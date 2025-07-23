// Common interface for array-like objects in NGFF
export interface ArrayLike {
  readonly shape: number[];
  readonly dtype: string;
  readonly chunks: number[];
  readonly path?: string;
}
