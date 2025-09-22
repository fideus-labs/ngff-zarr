// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { Multiscales } from "./multiscales.ts";
import type {
  PlateAcquisition,
  PlateColumn,
  PlateRow,
  PlateWell,
  WellImage,
} from "../schemas/index.ts";

export interface LRUCacheOptions {
  maxSize: number;
}

export class LRUCache<K, V> {
  private cache: Map<K, V>;
  private readonly maxSize: number;

  constructor(options: LRUCacheOptions) {
    this.cache = new Map();
    this.maxSize = options.maxSize;
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      // Update existing item
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove least recently used item
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, value);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }
}

export interface PlateMetadata {
  columns: PlateColumn[];
  rows: PlateRow[];
  wells: PlateWell[];
  version: string;
  acquisitions?: PlateAcquisition[] | undefined;
  field_count?: number | undefined;
  name?: string | undefined;
}

export interface WellMetadata {
  images: WellImage[];
  version: string;
}

export interface HCSPlateOptions {
  store: string | object;
  metadata: PlateMetadata;
  wellCacheSize?: number | undefined;
  imageCacheSize?: number | undefined;
}

export interface HCSWellOptions {
  store: string | object;
  wellPath: string;
  plateMetadata: PlateWell;
  wellGroupMetadata: WellMetadata;
  imageCacheSize?: number | undefined;
}

export class HCSWell {
  private readonly store: string | object;
  public readonly path: string;
  public readonly plateMetadata: PlateWell;
  public readonly metadata: WellMetadata;
  private readonly _images: LRUCache<string, Multiscales>;

  constructor(options: HCSWellOptions) {
    this.store = options.store;
    this.path = options.wellPath;
    this.plateMetadata = { ...options.plateMetadata };
    this.metadata = {
      images: [...options.wellGroupMetadata.images],
      version: options.wellGroupMetadata.version,
    };

    const imageCacheSize = options.imageCacheSize ?? 100;
    this._images = new LRUCache<string, Multiscales>({
      maxSize: imageCacheSize,
    });
  }

  get rowIndex(): number {
    return this.plateMetadata.rowIndex;
  }

  get columnIndex(): number {
    return this.plateMetadata.columnIndex;
  }

  get images(): WellImage[] {
    return [...this.metadata.images];
  }

  async getImage(fieldIndex: number = 0): Promise<Multiscales | null> {
    if (fieldIndex < 0 || fieldIndex >= this.metadata.images.length) {
      return null;
    }

    const imageMeta = this.metadata.images[fieldIndex];
    const imagePath = `${this.path}/${imageMeta.path}`;

    // Check cache first
    const cached = this._images.get(imagePath);
    if (cached) {
      return cached;
    }

    // Load the image using fromNgffZarr
    const { fromNgffZarr } = await import("../io/from_ngff_zarr.ts");

    let fullImagePath: string;
    if (typeof this.store === "string") {
      // If store is a path string, append the image path
      fullImagePath = `${this.store}/${this.path}/${imageMeta.path}`;
    } else {
      // For other store types, we need to handle differently
      // This would need to be adapted based on the specific store type
      throw new Error("Non-string store types not yet implemented for HCS");
    }

    try {
      const multiscales = await fromNgffZarr(fullImagePath);
      this._images.set(imagePath, multiscales);
      return multiscales;
    } catch (error) {
      console.error(`Failed to load image at ${fullImagePath}:`, error);
      return null;
    }
  }

  getImageByAcquisition(
    acquisitionId: number,
    fieldIndex: number = 0,
  ): Promise<Multiscales | null> {
    // Find images for the specified acquisition
    const acquisitionImages = this.metadata.images.filter(
      (img) => img.acquisition === acquisitionId,
    );

    if (fieldIndex < 0 || fieldIndex >= acquisitionImages.length) {
      return Promise.resolve(null);
    }

    // Find the actual image index in the full list
    const targetImage = acquisitionImages[fieldIndex];
    const actualIndex = this.metadata.images.indexOf(targetImage);

    return this.getImage(actualIndex);
  }

  static fromStore(
    store: string | object,
    wellPath: string,
    wellMetadata: PlateWell,
    imageCacheSize?: number | undefined,
  ): HCSWell {
    // Simplified implementation - we'll improve this when implementing the IO functions
    const wellGroupMetadata: WellMetadata = {
      images: [
        { path: "0", acquisition: 0 },
        { path: "1", acquisition: 0 },
      ],
      version: "0.4",
    };

    return new HCSWell({
      store,
      wellPath,
      plateMetadata: wellMetadata,
      wellGroupMetadata,
      imageCacheSize,
    });
  }
}

export class HCSPlate {
  private readonly store: string | object;
  public readonly metadata: PlateMetadata;
  private readonly _wells: LRUCache<string, HCSWell>;
  private readonly imageCacheSize: number | undefined;

  constructor(options: HCSPlateOptions) {
    this.store = options.store;
    this.metadata = {
      columns: [...options.metadata.columns],
      rows: [...options.metadata.rows],
      wells: [...options.metadata.wells],
      version: options.metadata.version,
      acquisitions: options.metadata.acquisitions
        ? [...options.metadata.acquisitions]
        : undefined,
      field_count: options.metadata.field_count,
      name: options.metadata.name,
    };
    this.imageCacheSize = options.imageCacheSize;

    const wellCacheSize = options.wellCacheSize ?? 500;
    this._wells = new LRUCache<string, HCSWell>({ maxSize: wellCacheSize });
  }

  get name(): string | undefined {
    return this.metadata.name;
  }

  get rows(): PlateRow[] {
    return [...this.metadata.rows];
  }

  get columns(): PlateColumn[] {
    return [...this.metadata.columns];
  }

  get wells(): PlateWell[] {
    return [...this.metadata.wells];
  }

  get acquisitions(): PlateAcquisition[] | undefined {
    return this.metadata.acquisitions
      ? [...this.metadata.acquisitions]
      : undefined;
  }

  get fieldCount(): number | undefined {
    return this.metadata.field_count;
  }

  getWell(rowName: string, columnName: string): HCSWell | null {
    const wellPath = `${rowName}/${columnName}`;

    // Check if well exists in metadata
    const wellMeta = this.metadata.wells.find((well) => well.path === wellPath);
    if (!wellMeta) {
      return null;
    }

    // Check cache first
    const cached = this._wells.get(wellPath);
    if (cached) {
      return cached;
    }

    // Load the well
    try {
      const well = HCSWell.fromStore(
        this.store,
        wellPath,
        wellMeta,
        this.imageCacheSize,
      );
      this._wells.set(wellPath, well);
      return well;
    } catch (error) {
      console.error(`Failed to load well at ${wellPath}:`, error);
      return null;
    }
  }

  getWellByIndices(rowIndex: number, columnIndex: number): HCSWell | null {
    if (
      rowIndex < 0 ||
      rowIndex >= this.metadata.rows.length ||
      columnIndex < 0 ||
      columnIndex >= this.metadata.columns.length
    ) {
      return null;
    }

    const rowName = this.metadata.rows[rowIndex].name;
    const columnName = this.metadata.columns[columnIndex].name;
    return this.getWell(rowName, columnName);
  }
}
