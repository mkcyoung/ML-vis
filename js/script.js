/** Adapted from: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?index=..%2F..index#2 */
import {MnistData} from './data.js';



async function showExamples(data) {
  // Get the examples
  const examples = data.nextTestBatch(20); //selects 20 images 
  const numExamples = examples.xs.shape[0]; //returns number of examples by capturing the "rows" of the tensor
  //console.log(numExamples)
  console.log(examples.xs)
  
  // Create a canvas element to render each example 
  // I wonder if I could/should do this with d3?
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {          //Tidy helps to prevent memory leakage
      // Reshape the image to 28x28 px
      return examples.xs
        //2-D tensor, specifies slicing from row, and taking out size of image: https://www.quora.com/How-does-tf-slice-work-in-TensorFlow
        .slice([i, 0], [1, examples.xs.shape[1]])
        //Reshapes to image size 
        .reshape([28, 28, 1]);
    });

    //Using D3 because...well...I like it.
    const div = d3.select("#load-view").append("div")

    //Create canvas element
    const canvas = div.append('canvas')
        .attr("class","preview")
        .attr("width",28)
        .attr("height",28)
        .style("margin","4px")
        .node();
    //Convert tensors to canvas images
    await tf.browser.toPixels(imageTensor, canvas); //Draws tensor of pixel values to byte array or canvas in this case

    //Draw canvases to div
    div.node().appendChild(canvas);

    //Cleans up tensor, again a memory thing.
    imageTensor.dispose(); 
  }
}

async function run() {  
  const data = new MnistData();
  await data.load();
  await showExamples(data);
}

document.addEventListener('DOMContentLoaded', run);