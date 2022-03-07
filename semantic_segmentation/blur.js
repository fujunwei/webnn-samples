const segWGSL = `
[[block]] struct SegMap {
  labels: array<u32>;
};

[[block]] struct SegMapSize {
  width: u32;
  height: u32;
};

[[group(0), binding(0)]] var samp : sampler;
[[group(0), binding(1)]] var<uniform> segmapSize : SegMapSize;
[[group(0), binding(2)]] var<storage, read> segmap : SegMap;
[[group(0), binding(3)]] var inputTex : texture_2d<f32>;
[[group(0), binding(4)]] var backgroundTex : texture_2d<f32>;
[[group(0), binding(5)]] var outputTex : texture_storage_2d<rgba8unorm, write>;

[[stage(compute), workgroup_size(8, 8)]]
fn main([[builtin(global_invocation_id)]] globalID : vec3<u32>) {
  let dims : vec2<i32> = textureDimensions(inputTex, 0);
  
  var input = textureSampleLevel(inputTex, samp, vec2<f32>(globalID.xy) / vec2<f32>(dims), 0.0).rgb;
  var background = textureSampleLevel(backgroundTex, samp, vec2<f32>(globalID.xy) / vec2<f32>(dims), 0.0).rgb;
  var green : vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
  var segmapX : u32 = u32(floor(f32(globalID.x) / f32(dims.x) * f32(segmapSize.width)));
  var segmapY : u32 = u32(floor(f32(globalID.y) / f32(dims.y) * f32(segmapSize.height)));
  var segmapIndex = segmapX + segmapY * segmapSize.height;
  if (segmap.labels[segmapIndex] == 0u) {
    textureStore(outputTex, vec2<i32>(globalID.xy), vec4<f32>(background, 1.0));
  } else {
    textureStore(outputTex, vec2<i32>(globalID.xy), vec4<f32>(input, 1.0));
  }
}
`

const blurWGSL = `
[[block]] struct Params {
  filterDim : u32;
  blockDim : u32;
};

[[group(0), binding(0)]] var samp : sampler;
[[group(0), binding(1)]] var<uniform> params : Params;
[[group(1), binding(1)]] var inputTex : texture_2d<f32>;
[[group(1), binding(2)]] var outputTex : texture_storage_2d<rgba8unorm, write>;

[[block]] struct Flip {
  value : u32;
};
[[group(1), binding(3)]] var<uniform> flip : Flip;

// This shader blurs the input texture in one direction, depending on whether
// |flip.value| is 0 or 1.
// It does so by running (128 / 4) threads per workgroup to load 128
// texels into 4 rows of shared memory. Each thread loads a
// 4 x 4 block of texels to take advantage of the texture sampling
// hardware.
// Then, each thread computes the blur result by averaging the adjacent texel values
// in shared memory.
// Because we're operating on a subset of the texture, we cannot compute all of the
// results since not all of the neighbors are available in shared memory.
// Specifically, with 128 x 128 tiles, we can only compute and write out
// square blocks of size 128 - (filterSize - 1). We compute the number of blocks
// needed in Javascript and dispatch that amount.

var<workgroup> tile : array<array<vec3<f32>, 128>, 4>;

[[stage(compute), workgroup_size(32, 1, 1)]]
fn main(
  [[builtin(workgroup_id)]] WorkGroupID : vec3<u32>,
  [[builtin(local_invocation_id)]] LocalInvocationID : vec3<u32>
) {
  let filterOffset : u32 = (params.filterDim - 1u) / 2u;
  let dims : vec2<i32> = textureDimensions(inputTex, 0);

  let baseIndex = vec2<i32>(
    WorkGroupID.xy * vec2<u32>(params.blockDim, 4u) +
    LocalInvocationID.xy * vec2<u32>(4u, 1u)
  ) - vec2<i32>(i32(filterOffset), 0);

  for (var r : u32 = 0u; r < 4u; r = r + 1u) {
    for (var c : u32 = 0u; c < 4u; c = c + 1u) {
      var loadIndex = baseIndex + vec2<i32>(i32(c), i32(r));
      if (flip.value != 0u) {
        loadIndex = loadIndex.yx;
      }

      tile[r][4u * LocalInvocationID.x + c] =
        textureSampleLevel(inputTex, samp,
          (vec2<f32>(loadIndex) + vec2<f32>(0.25, 0.25)) / vec2<f32>(dims), 0.0).rgb;
    }
  }

  workgroupBarrier();

  for (var r : u32 = 0u; r < 4u; r = r + 1u) {
    for (var c : u32 = 0u; c < 4u; c = c + 1u) {
      var writeIndex = baseIndex + vec2<i32>(i32(c), i32(r));
      if (flip.value != 0u) {
        writeIndex = writeIndex.yx;
      }

      let center : u32 = 4u * LocalInvocationID.x + c;
      if (center >= filterOffset &&
          center < 128u - filterOffset &&
          all(writeIndex < dims)) {
        var acc : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        for (var f : u32 = 0u; f < params.filterDim; f = f + 1u) {
          var i : u32 = center + f - filterOffset;
          acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
        }
        textureStore(outputTex, writeIndex, vec4<f32>(acc, 1.0));
      }
    }
  }
}
`;

const fullscreenTexturedQuadWGSL = `
[[group(0), binding(0)]] var mySampler : sampler;
[[group(0), binding(1)]] var myTexture : texture_2d<f32>;

struct VertexOutput {
  [[builtin(position)]] Position : vec4<f32>;
  [[location(0)]] fragUV : vec2<f32>;
};

[[stage(vertex)]]
fn vert_main([[builtin(vertex_index)]] VertexIndex : u32) -> VertexOutput {
  var pos = array<vec2<f32>, 6>(
      vec2<f32>( 1.0,  1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0,  1.0));

  var uv = array<vec2<f32>, 6>(
      vec2<f32>(1.0, 0.0),
      vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(0.0, 0.0));

  var output : VertexOutput;
  output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}

[[stage(fragment)]]
fn frag_main([[location(0)]] fragUV : vec2<f32>) -> [[location(0)]] vec4<f32> {
  return textureSample(myTexture, mySampler, fragUV);
}
`;

// Contants from the blur.wgsl shader.
const tileDim = 128;
const batch = [4, 4];

export async function drawOutput(img, canvas, segmapBuffer, device) {
  const adapter = await navigator.gpu.requestAdapter();

  const context = canvas.getContext('webgpu');

  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [
    canvas.clientWidth * devicePixelRatio,
    canvas.clientHeight * devicePixelRatio,
  ];
  const presentationFormat = context.getPreferredFormat(adapter);

  context.configure({
    device,
    format: presentationFormat,
    size: presentationSize,
  });

  const segPipeline = device.createComputePipeline({
    compute: {
      module: device.createShaderModule({
        code: segWGSL,
      }),
      entryPoint: 'main',
    },
  });

  const blurPipeline = device.createComputePipeline({
    compute: {
      module: device.createShaderModule({
        code: blurWGSL,
      }),
      entryPoint: 'main',
    },
  });

  const fullscreenQuadPipeline = device.createRenderPipeline({
    vertex: {
      module: device.createShaderModule({
        code: fullscreenTexturedQuadWGSL,
      }),
      entryPoint: 'vert_main',
    },
    fragment: {
      module: device.createShaderModule({
        code: fullscreenTexturedQuadWGSL,
      }),
      entryPoint: 'frag_main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  await img.decode();
  const imageBitmap = await createImageBitmap(img);
  const [srcWidth, srcHeight] = [imageBitmap.width, imageBitmap.height];
  // const [srcWidth, srcHeight] = [513, 513];
  // const inputCanvas = document.createElement('canvas');
  // inputCanvas.width = srcWidth;
  // inputCanvas.height = srcHeight;
  // const canvasContext = inputCanvas.getContext('2d');
  // canvasContext.drawImage(img, 0, 0, srcWidth, srcHeight);

  canvas.width = srcWidth;
  canvas.height = srcHeight;
  const cubeTexture = device.createTexture({
    size: [srcWidth, srcHeight, 1],
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: cubeTexture },
    [ srcWidth, srcHeight]
  );

  const textures = [0, 1, 2].map(() => {
    return device.createTexture({
      size: {
        width: srcWidth,
        height: srcHeight,
      },
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING,
    });
  });

  const buffer0 = (() => {
    const buffer = device.createBuffer({
      size: 4,
      mappedAtCreation: true,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Uint32Array(buffer.getMappedRange())[0] = 0;
    buffer.unmap();
    return buffer;
  })();

  const buffer1 = (() => {
    const buffer = device.createBuffer({
      size: 4,
      mappedAtCreation: true,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Uint32Array(buffer.getMappedRange())[0] = 1;
    buffer.unmap();
    return buffer;
  })();

  const segmapSizeBuffer = (() => {
    const buffer = device.createBuffer({
      size: 2 * Uint32Array.BYTES_PER_ELEMENT,
      mappedAtCreation: true,
      usage: GPUBufferUsage.UNIFORM,
    });
    new Uint32Array(buffer.getMappedRange()).set([513, 513]);
    buffer.unmap();
    return buffer;
  })();

  const blurParamsBuffer = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });

  const computeSegBindGroup = device.createBindGroup({
    layout: segPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: sampler,
      },
      {
        binding: 1,
        resource: {
          buffer: segmapSizeBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: segmapBuffer,
        },
      },
      {
        binding: 3,
        resource: cubeTexture.createView(),
      },
      {
        binding: 4,
        resource: textures[0].createView(),
      },
      {
        binding: 5,
        resource: textures[2].createView(),
      },
    ],
  });

  const computeConstants = device.createBindGroup({
    layout: blurPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: sampler,
      },
      {
        binding: 1,
        resource: {
          buffer: blurParamsBuffer,
        },
      },
    ],
  });

  const computeBindGroup0 = device.createBindGroup({
    layout: blurPipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 1,
        resource: cubeTexture.createView(),
      },
      {
        binding: 2,
        resource: textures[0].createView(),
      },
      {
        binding: 3,
        resource: {
          buffer: buffer0,
        },
      },
    ],
  });

  const computeBindGroup1 = device.createBindGroup({
    layout: blurPipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 1,
        resource: textures[0].createView(),
      },
      {
        binding: 2,
        resource: textures[1].createView(),
      },
      {
        binding: 3,
        resource: {
          buffer: buffer1,
        },
      },
    ],
  });

  const computeBindGroup2 = device.createBindGroup({
    layout: blurPipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 1,
        resource: textures[1].createView(),
      },
      {
        binding: 2,
        resource: textures[0].createView(),
      },
      {
        binding: 3,
        resource: {
          buffer: buffer0,
        },
      },
    ],
  });

  const showResultBindGroup = device.createBindGroup({
    layout: fullscreenQuadPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: sampler,
      },
      {
        binding: 1,
        resource: textures[2].createView(),
      },
    ],
  });

  const settings = {
    filterSize: 50,
    iterations: 2,
  };

  let blockDim = tileDim - (settings.filterSize - 1);
  device.queue.writeBuffer(
    blurParamsBuffer,
    0,
    new Uint32Array([settings.filterSize, blockDim])
  );
  if (!canvas) return;

  const commandEncoder = device.createCommandEncoder();

  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(blurPipeline);
  computePass.setBindGroup(0, computeConstants);

  computePass.setBindGroup(1, computeBindGroup0);
  computePass.dispatch(
    Math.ceil(srcWidth / blockDim),
    Math.ceil(srcHeight / batch[1])
  );

  computePass.setBindGroup(1, computeBindGroup1);
  computePass.dispatch(
    Math.ceil(srcHeight / blockDim),
    Math.ceil(srcWidth / batch[1])
  );

  for (let i = 0; i < settings.iterations - 1; ++i) {
    computePass.setBindGroup(1, computeBindGroup2);
    computePass.dispatch(
      Math.ceil(srcWidth / blockDim),
      Math.ceil(srcHeight / batch[1])
    );

    computePass.setBindGroup(1, computeBindGroup1);
    computePass.dispatch(
      Math.ceil(srcHeight / blockDim),
      Math.ceil(srcWidth / batch[1])
    );
  }

  computePass.setPipeline(segPipeline);
  computePass.setBindGroup(0, computeSegBindGroup);
  computePass.dispatch(
    Math.ceil(srcWidth / 8),
    Math.ceil(srcHeight / 8)
  );

  computePass.endPass();

  const passEncoder = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        storeOp: 'store',
      },
    ],
  });

  passEncoder.setPipeline(fullscreenQuadPipeline);
  passEncoder.setBindGroup(0, showResultBindGroup);
  passEncoder.draw(6, 1, 0, 0);
  passEncoder.endPass();
  device.queue.submit([commandEncoder.finish()]);
}