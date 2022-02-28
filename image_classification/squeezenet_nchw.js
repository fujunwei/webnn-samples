'use strict';

import {buildConstantByNpy, sizeOfShape} from '../common/utils.js';

// SqueezeNet 1.1 model with 'nchw' input layout
export class SqueezeNetNchw {
  constructor() {
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/squeezenet1.1_nchw/weights/';
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
    };
    this.outputDimensions = [1, 1000];
    this.inputSizeInBytes_ = sizeOfShape(this.inputOptions.inputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.outputSizeInBytes_ = sizeOfShape(this.outputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.inputGPUBuffers_ = [];
    this.outputGPUBuffer_ = null;
  }

  async buildConv_(input, name, options = {}) {
    const prefix = this.weightsUrl_ + 'squeezenet0_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = await buildConstantByNpy(this.device_, this.builder_, weightsName);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.device_, this.builder_, biasName);
    options.bias = bias;
    options.activation = this.builder_.relu();
    return this.builder_.conv2d(input, weights, options);
  }

  async buildFire_(input, convName, conv1x1Name, conv3x3Name) {
    const conv = await this.buildConv_(input, convName);
    const conv1x1 = await this.buildConv_(conv, conv1x1Name);
    const conv3x3 = await this.buildConv_(
        conv, conv3x3Name, {padding: [1, 1, 1, 1]});
    return this.builder_.concat([conv1x1, conv3x3], 1);
  }

  async load(devicePreference) {
    const adaptor = await navigator.gpu.requestAdapter();
    this.device_ = await adaptor.requestDevice();
    const context = navigator.ml.createContext(this.device_);
    this.builder_ = new MLGraphBuilder(context);
    const data = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const conv0 = await this.buildConv_(data, 'conv0', {strides: [2, 2]});
    const pool0 = this.builder_.maxPool2d(
        conv0, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire0 = await this.buildFire_(pool0, 'conv1', 'conv2', 'conv3');
    const fire1 = await this.buildFire_(fire0, 'conv4', 'conv5', 'conv6');
    const pool1 = this.builder_.maxPool2d(
        fire1, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire2 = await this.buildFire_(pool1, 'conv7', 'conv8', 'conv9');
    const fire3 = await this.buildFire_(fire2, 'conv10', 'conv11', 'conv12');
    const pool2 = this.builder_.maxPool2d(
        fire3, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire4 = await this.buildFire_(pool2, 'conv13', 'conv14', 'conv15');
    const fire5 = await this.buildFire_(fire4, 'conv16', 'conv17', 'conv18');
    const fire6 = await this.buildFire_(fire5, 'conv19', 'conv20', 'conv21');
    const fire7 = await this.buildFire_(fire6, 'conv22', 'conv23', 'conv24');
    const conv25 = await this.buildConv_(fire7, 'conv25');
    const pool3 = this.builder_.averagePool2d(
        conv25, {windowDimensions: [13, 13], strides: [13, 13]});
    const reshape0 = this.builder_.reshape(pool3, [1, -1]);
    return this.builder_.softmax(reshape0);
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
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
    let inputGPUBuffer;
    if (this.inputGPUBuffers_.length) {
      inputGPUBuffer = this.inputGPUBuffers_.pop();
    } else {
      inputGPUBuffer = this.device_.createBuffer({
        size: this.inputSizeInBytes_,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
      });
      await inputGPUBuffer.mapAsync(GPUMapMode.WRITE);
    }
    new Float32Array(inputGPUBuffer.getMappedRange()).set(inputBuffer);
    inputGPUBuffer.unmap();
    this.graph_.compute({'input': {resource: inputGPUBuffer}}, {'output': {resource: this.outputGPUBuffer_}});
    inputGPUBuffer.mapAsync(GPUMapMode.WRITE).then(() => {
      this.inputGPUBuffers_.push(inputGPUBuffer);
    });
    await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    outputBuffer.set(new Float32Array(this.outputGPUBuffer_.getMappedRange()));
    this.outputGPUBuffer_.unmap();
  }
}
