'use strict';

import {buildConstantByNpy, sizeOfShape} from '../common/utils.js';

// SqueezeNet 1.0 model with 'nhwc' layout
export class SqueezeNetNhwc {
  constructor() {
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/squeezenet1.0_nhwc/weights/';
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      inputLayout: 'nhwc',
      labelUrl: './labels/labels1001.txt',
      inputDimensions: [1, 224, 224, 3],
    };
    this.outputDimensions = [1, 1001];
    this.inputSizeInBytes_ = sizeOfShape(this.inputOptions.inputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.outputSizeInBytes_ = sizeOfShape(this.outputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.inputGPUBuffer_ = null;
    this.outputGPUBuffer_ = null;
  }

  async buildConv_(input, name, options = {}) {
    const prefix = this.weightsUrl_ + name;
    const weightsName = prefix + '_kernel.npy';
    const weights = await buildConstantByNpy(this.device_, this.builder_, weightsName);
    const biasName = prefix + '_Conv2D_bias.npy';
    const bias = await buildConstantByNpy(this.device_, this.builder_, biasName);
    options.inputLayout = 'nhwc';
    options.filterLayout = 'ohwi';
    options.bias = bias;
    options.activation = this.builder_.relu();
    return this.builder_.conv2d(input, weights, options);
  }

  async buildFire_(input, name) {
    const convSqueeze = await this.buildConv_(input, name + '_squeeze');
    const convE1x1 = await this.buildConv_(convSqueeze, name + '_e1x1');
    const convE3x3 = await this.buildConv_(
        convSqueeze, name + '_e3x3', {padding: [1, 1, 1, 1]});
    return this.builder_.concat([convE1x1, convE3x3], 3);
  }

  async load(devicePreference) {
    const adaptor = await navigator.gpu.requestAdapter();
    this.device_ = await adaptor.requestDevice();
    const context = navigator.ml.createContext(this.device_);
    this.builder_ = new MLGraphBuilder(context);
    const strides = [2, 2];
    const layout = 'nhwc';
    const placeholder = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const conv1 = await this.buildConv_(
        placeholder, 'conv1', {strides, autoPad: 'same-upper'});
    const maxpool1 = this.builder_.maxPool2d(
        conv1, {windowDimensions: [3, 3], strides, layout});
    const fire2 = await this.buildFire_(maxpool1, 'fire2');
    const fire3 = await this.buildFire_(fire2, 'fire3');
    const fire4 = await this.buildFire_(fire3, 'fire4');
    const maxpool4 = this.builder_.maxPool2d(
        fire4, {windowDimensions: [3, 3], strides, layout});
    const fire5 = await this.buildFire_(maxpool4, 'fire5');
    const fire6 = await this.buildFire_(fire5, 'fire6');
    const fire7 = await this.buildFire_(fire6, 'fire7');
    const fire8 = await this.buildFire_(fire7, 'fire8');
    const maxpool8 = this.builder_.maxPool2d(
        fire8, {windowDimensions: [3, 3], strides, layout});
    const fire9 = await this.buildFire_(maxpool8, 'fire9');
    const conv10 = await this.buildConv_(fire9, 'conv10');
    const averagePool2d = this.builder_.averagePool2d(
        conv10, {windowDimensions: [13, 13], layout});
    const reshape = this.builder_.reshape(averagePool2d, [1, -1]);
    return this.builder_.softmax(reshape);
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
    this.inputGPUBuffer_ = this.device_.createBuffer({size: this.inputSizeInBytes_, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    this.outputGPUBuffer_ = this.device_.createBuffer({size: this.outputSizeInBytes_, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  async compute(inputBuffer, outputBuffer) {
    this.device_.queue.writeBuffer(this.inputGPUBuffer_, 0, inputBuffer.buffer, 0, this.inputSizeInBytes_);
    this.graph_.compute({'input': {resource: this.inputGPUBuffer_}}, {'output': {resource: this.outputGPUBuffer_}});
    await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    outputBuffer.set(new Float32Array(this.outputGPUBuffer_.getMappedRange()));
    this.outputGPUBuffer_.unmap();
  }
}
