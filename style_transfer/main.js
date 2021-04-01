'use strict';

import {FastStyleTransferNet} from './fast_style_transfer_net.js';
import {showProgressComponent, readyShowResultComponents} from './ui.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/content-images/travelspace.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelId = 'starry-night';
let isFirstTimeLoad = true;
let isModelChanged = false;
let shouldStopFrame = false;
let inputType = 'image';
let fastStyleTransferNet;
let stream = null;
let loadTime = 0;
let compileTime = 0;
let computeTime = 0;

$(document).ready(() => {
  $('.icdisplay').hide();
  $('.badge').html(modelId);
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  shouldStopFrame = true;
  if (stream !== null) {
    stopCamera();
  }
  inputType = 'image';
  $('.shoulddisplay').hide();
  await main();
});

$('#gallery .gallery-image').hover((e) => {
  const id = $(e.target).attr('id');
  const modelName = $('#' + id).attr('title');
  $('.badge').html(modelName);
}, () => {
  const modelName = $(`#${modelId}`).attr('title');
  $('.badge').html(modelName);
});

$('#imageFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#feedElement').on('load', async () => {
      await main();
    });
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('.shoulddisplay').hide();
  await main();
});

// Click handler to do inference with switched <img> element
async function handleImageSwitch(e) {
  const newModelId = $(e.target).attr('id');
  if (newModelId !== modelId) {
    shouldStopFrame = true;
    isModelChanged = true;
    modelId = newModelId;
    const modelName = $(`#${modelId}`).attr('title');
    $('.badge').html(modelName);
    $('#gallery .gallery-item').removeClass('hl');
    $(e.target).parent().addClass('hl');
    await main();
  }
}

async function getMediaStream() {
  // Support 'user' facing mode at present
  const constraints = {audio: false, video: {facingMode: 'user'}};
  stream = await navigator.mediaDevices.getUserMedia(constraints);
}

function stopCamera() {
  stream.getTracks().forEach((track) => {
    if (track.readyState === 'live' && track.kind === 'video') {
      track.stop();
    }
  });
}

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  const inputBuffer = await fastStyleTransferNet.preprocess(camElement);
  console.log('- Computing... ');
  const start = performance.now();
  const outputs = await fastStyleTransferNet.compute(inputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  camElement.width = camElement.videoWidth;
  camElement.height = camElement.videoHeight;
  drawInput(camElement, 'camInCanvas');
  showPerfResult();
  await drawOutput(outputs, 'camInCanvas', 'camOutCanvas');
  if (!shouldStopFrame) {
    requestAnimationFrame(renderCamStream);
  }
}

function drawInput(srcElement, canvasId) {
  const inputCanvas = document.getElementById(canvasId);
  const resizeRatio = Math.max(
      Math.max(srcElement.width / maxWidth, srcElement.height / maxHeight), 1);
  const scaledWidth = Math.floor(srcElement.width / resizeRatio);
  const scaledHeight = Math.floor(srcElement.height / resizeRatio);
  inputCanvas.height = scaledHeight;
  inputCanvas.width = scaledWidth;
  const ctx = inputCanvas.getContext('2d');
  ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);
}

async function drawOutput(outputs, inCanvasId, outCanvasId) {
  const outputTensor = outputs.output.buffer;
  const outputSize = outputs.output.dimensions;
  const height = outputSize[2];
  const width = outputSize[3];
  const mean = [1, 1, 1, 1];
  const offset = [0, 0, 0, 0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const a = 255;

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const r = outputTensor[i] * mean[0] + offset[0];
    const g = outputTensor[i + height * width] * mean[1] + offset[1];
    const b = outputTensor[i + height * width * 2] * mean[2] + offset[2];
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }

  const imageData = new ImageData(bytes, width, height);
  const outCanvas = document.createElement('canvas');
  const outCtx = outCanvas.getContext('2d');
  outCanvas.width = width;
  outCanvas.height = height;
  outCtx.putImageData(imageData, 0, 0, 0, 0, outCanvas.width, outCanvas.height);

  const inputCanvas = document.getElementById(inCanvasId);
  const outputCanvas = document.getElementById(outCanvasId);
  outputCanvas.width = inputCanvas.width;
  outputCanvas.height = inputCanvas.height;
  const ctx = outputCanvas.getContext('2d');
  ctx.drawImage(outCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
}

function showPerfResult() {
  $('#loadTime').html(`${loadTime} ms`);
  $('#compileTime').html(`${compileTime} ms`);
  $('#computeTime').html(`${computeTime} ms`);
}

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

export async function main() {
  try {
    let start;
    // Only do load() and compile() when page first time loads and
    // there's new model choosed
    if (isFirstTimeLoad || isModelChanged) {
      if (fastStyleTransferNet !== undefined) {
        // Call dispose() to and avoid memory leak
        fastStyleTransferNet.dispose();
      }
      fastStyleTransferNet = new FastStyleTransferNet();
      isFirstTimeLoad = false;
      isModelChanged = false;
      console.log(`- Model ID: ${modelId} -`);
      // UI shows loading model progress
      await showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      start = performance.now();
      await fastStyleTransferNet.load(modelId);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows compiling model progress
      await showProgressComponent('done', 'current', 'pending');
      console.log('- Compiling... ');
      start = performance.now();
      await fastStyleTransferNet.compile();
      compileTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${compileTime} ms.`);
    }
    // UI shows inferencing progress
    await showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer = await fastStyleTransferNet.preprocess(imgElement);
      console.log('- Computing... ');
      start = performance.now();
      const outputs = await fastStyleTransferNet.compute(inputBuffer);
      computeTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${computeTime} ms.`);
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
      drawInput(imgElement, 'inputCanvas');
      await drawOutput(outputs, 'inputCanvas', 'outputCanvas');
      showPerfResult();
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
      shouldStopFrame = false;
      camElement.onloadedmediadata = await renderCamStream();
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
    $('#gallery .gallery-image').on('click', handleImageSwitch);
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
