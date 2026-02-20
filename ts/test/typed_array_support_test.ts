import { assertEquals } from "@std/assert";
import { toNgffImage } from "../src/io/to_ngff_image.ts";
import { toMultiscales } from "../src/process/to_multiscales-node.ts";
import { Methods } from "../src/types/methods.ts";

Deno.test("support Int8Array", async () => {
  const data = new Int8Array(256 * 256);
  for (let i = 0; i < data.length; i++) {
    data[i] = (i % 128) - 64;
  }

  const image = await toNgffImage(data, {
    dims: ["y", "x"],
    shape: [256, 256],
  });

  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_BIN_SHRINK,
  });

  assertEquals(multiscales.images.length, 2);
  assertEquals(multiscales.images[1].data.shape[0], 128);
  assertEquals(multiscales.images[1].data.shape[1], 128);
});

Deno.test("support Uint32Array", async () => {
  const data = new Uint32Array(128 * 128);
  for (let i = 0; i < data.length; i++) {
    data[i] = i * 1000;
  }

  const image = await toNgffImage(data, {
    dims: ["y", "x"],
    shape: [128, 128],
  });

  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_BIN_SHRINK,
  });

  assertEquals(multiscales.images.length, 2);
  assertEquals(multiscales.images[1].data.shape[0], 64);
  assertEquals(multiscales.images[1].data.shape[1], 64);
});

Deno.test("support Int32Array", async () => {
  const data = new Int32Array(128 * 128);
  for (let i = 0; i < data.length; i++) {
    data[i] = i - 8192;
  }

  const image = await toNgffImage(data, {
    dims: ["y", "x"],
    shape: [128, 128],
  });

  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_BIN_SHRINK,
  });

  assertEquals(multiscales.images.length, 2);
  assertEquals(multiscales.images[1].data.shape[0], 64);
  assertEquals(multiscales.images[1].data.shape[1], 64);
});

Deno.test("support Float64Array", async () => {
  const data = new Float64Array(128 * 128);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.sin(i / 100.0) * 1000.0;
  }

  const image = await toNgffImage(data, {
    dims: ["y", "x"],
    shape: [128, 128],
  });

  const multiscales = await toMultiscales(image, {
    scaleFactors: [2],
    method: Methods.ITKWASM_GAUSSIAN,
  });

  assertEquals(multiscales.images.length, 2);
  assertEquals(multiscales.images[1].data.shape[0], 64);
  assertEquals(multiscales.images[1].data.shape[1], 64);
});

console.log("✅ All TypedArray support tests completed!");
