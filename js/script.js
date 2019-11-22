/** Adapted from: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?index=..%2F..index#2 */
import {MnistData} from './data.js';



async function showExamples(data,container_id,num_images) {
  // Get the examples
  let examples = null;
  if(num_images>1){
    examples = data.nextTestBatch(num_images); //selects images
  }
  else{
    examples = data;
  }
  const numExamples = examples.xs.shape[0]; //returns number of examples by capturing the "rows" of the tensor
  const labels = Array.from(examples.labels.argMax(1).dataSync()); //Returns list of labels 
  //console.log("examples.xs:",examples.xs);
  //console.log(examples.labels.arraySync())
  
  // Create a canvas element (with d3) to render each example 
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
    const div = d3.select(`#${container_id}`).append("div")
        .attr("class","pre-view-container");

    //Add label to div 
    div.append("text")
        .text("Labeled: "+labels[i]);

    //Create canvas element
    const canvas = div.append('canvas')
        .attr("class","pre-view")
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



/** My run function */
async function run() {  
  const data = new MnistData();
  await data.load();
  await showExamples(data,"load-view",13);

  const model = getModel();
  //Visualizing model summary
  let container = d3.select("#model-summary");
  //tfvis.show.modelSummary(container.node(), model);
  tfvis.show.layer(container.node(), model);
  //tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  
  await train(model, data);

  //Prediction and evaluation
  //await showAccuracy(model, data);
  //await showConfusion(model, data);

  await showPrediction(model,data);
}

document.addEventListener('DOMContentLoaded', run);



/** Experimenting with different model types */

function getModel2() {
  const input = tf.layers.flatten( { inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS] } )
  const denseLayer = tf.layers.dense({units: 1});
  const activationLayer = tf.layers.activation({activation: 'relu6'});

  // Obtain the output symbolic tensors by applying the layers in order.
  const denseOutput = denseLayer.apply(input);
  const activationOutput = activationLayer.apply(denseOutput);

  // Create the model based on the inputs.
  const model = tf.model({
      inputs: input,
      outputs: [denseOutput, activationOutput]
  });

  // Collect both outputs and print separately.
  const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
  denseOut.print();
  activationOut.print();



}




/** Model architecture 
 * I want to eventually have some controls that adjust parameters within this.
*/

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    
    /** Standard Artificial Neural Network 
     * Want tp create controls for
     * Activation fn
     * # of layers
     * # of units in a layer
    */

    //A Flatten layer flattens each batch in its inputs to 1D (making the output 2D).
    model.add( tf.layers.flatten( { inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS] } ) ) 
    //model.add(tf.layers.dense({inputShape: [784], units: 10, activation: 'softmax'} ) )


    //Creates a dense (fully connected) layer.
        // This layer implements the operation: output = activation(dot(input, kernel) + bias)
        // activation is the element-wise activation function passed as the activation argument.
        // kernel is a weights matrix created by the layer.
        // bias is a bias vector created by the layer (only applicable if useBias is true).
    //Allow user selected different activation functions here + change number of layers
    //model.add( tf.layers.dense( { units: 10, activation: 'sigmoid' } ) )
    //model.add( tf.layers.dense( { units: 50, activation: 'relu' } ) )

    //Softmax layer at end
    model.add( tf.layers.dense( { units: 10, activation: 'softmax'} ) ) //'softmax




    
    /** Convolutional Neural Network */

    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    // model.add(tf.layers.conv2d({
    //   inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    //   kernelSize: 5,
    //   filters: 8,
    //   strides: 1,
    //   activation: 'relu',
    //   kernelInitializer: 'varianceScaling'
    // }));

    // // The MaxPooling layer acts as a sort of downsampling using max values
    // // in a region instead of averaging.  
    // model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // // Repeat another conv2d + maxPooling stack. 
    // // Note that we have more filters in the convolution.
    // model.add(tf.layers.conv2d({
    //   kernelSize: 5,
    //   filters: 16,
    //   strides: 1,
    //   activation: 'relu',
    //   kernelInitializer: 'varianceScaling'
    // }));
    // model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // // it for input into our last layer. This is common practice when feeding
    // // higher dimensional data to a final classification output layer.
    // model.add(tf.layers.flatten());
  
    // // Our last layer is a dense layer which has 10 output units, one for each
    // // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    // const NUM_OUTPUT_CLASSES = 10;
    // model.add(tf.layers.dense({
    //   units: NUM_OUTPUT_CLASSES,
    //   kernelInitializer: 'varianceScaling',
    //   activation: 'softmax'
    // }));
  
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    // Constructs a tf.AdamOptimizer that uses the Adam algorithm. See https://arxiv.org/abs/1412.6980
    let learning_rate = 0.05;
    const optimizer = tf.train.adam(learning_rate);
    model.compile({
      optimizer: optimizer,
    //   As the name implies this is used when the output 
    //   of our model is a probability distribution. categoricalCrossentropy measures 
    //   the error between the probability distribution generated by the last layer of our 
    //   model and the probability distribution given by our true label.
      loss: 'categoricalCrossentropy', 
      // loss: 'meanSquaredError',
      metrics: ['accuracy'], //Might be able to add precision or recall here too?
    });
  
    return model;
  }




  /** Training with the model */

  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];

    //Visualizing these training metrics

    // const container = {
    //   name: 'Model Training', styles: { height: '1000px' }
    // };

    // Set my own container
    let container = d3.select("#training-view");
    //Inserts the tfvis metric graphs into model-view div
    const fitCallbacks = tfvis.show.fitCallbacks(container.node(), metrics);
    //const fitCallbacks = onEpochEnd(epoch, showLayer(model));
    
    //Allow user to define these
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    //Sets up training tensors
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        //d.xs,
        d.labels
      ];
    });
  
    //Sets up test tensors
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        //d.xs,
        d.labels
      ];
    });
  


    //Setting up layer divs to be using in callback
    const div = d3.select(`#layer`).append("div")
        .attr("class","weight-container");

    const canvas = div.append('canvas')
      .attr("class","weight")
      .attr("width",30)
      .attr("height",30)
      .style("margin","4px")
      .node();


    // model.fit starts the training loop
    let NUM_EPOCHS = 5;


    const trainLogs = [];
    const lossContainer = document.getElementById('training-view');
    const beginMs = performance.now();
    
    //return model.fit(trainXs, trainYs, {
    const history = await model.fit(trainXs,trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: NUM_EPOCHS,
      shuffle: true,
      callbacks: {
      
          onEpochEnd: async (epoch, logs) => {
            // Plot the loss and accuracy values at the end of every training epoch.
            // const secPerEpoch =
            //     (performance.now() - beginMs) / (1000 * (epoch + 1));
            // ui.status(`Training model... Approximately ${
            //     secPerEpoch.toFixed(4)} seconds per epoch`)
            trainLogs.push(logs);
            tfvis.show.history(lossContainer, trainLogs, metrics)
            // calculateAndDrawConfusionMatrix(model, xTest, yTest);
            

            var ctx = canvas.getContext('2d');
            var ImageData = ctx.createImageData(28, 28);

            //showLayer(model,ctx,ImageData)
            
            
          },
        }
    });
    return model


  }


//Evaluating the model / making predictions

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

//Makes predictions once model is trained - new data there aren't existing labels
function doPrediction(model, data, testDataSize) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  //This selects a random examples from the data
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs);//.argMax([-1]);

  //This shows me the output of softmax along with actual label
  //console.log(model.predict(testxs).dataSync(),testData.labels.argMax([-1]).dataSync())

  testxs.dispose();
  return [preds, labels, testData];
}



/**
 * This function shows a prediction example and its corrresponding softmax histogram
 * @param {the training model} model 
 * @param {the input data} data 
 */
async function showPrediction(model,data){
  //Generates 
  let [preds,labels,testData] = doPrediction(model, data,1);

  //Makes a barchart showing a prediction for a single example
  let myBarChart = d3.select("#histo").node();
  let barchartData = Array.from(preds.dataSync()).map((d, i) => {
    return { index: i, value: d }
  })
  tfvis.render.barchart(myBarChart, barchartData,  { width: 750, height: 200, fontSize:18 })

  //Shows the predicted example
  showExamples(testData,"input",1)

  //Shows layers of first hidden layer
  showLayer(model)


}

/**
 * This function visualizes nodes in layers
 * @param {layer data} data 
 */
async function showLayer(model,ctx,ImageData){

  //First step is to retrieve all of the weights from the designated layer...I think
  let layer = model.getLayer('dense_Dense1')
  let surface = d3.select("#layer").node();

  // .output does something which might be useful
  let layerOutput = layer.output
  //console.log(layerOutput)

  //Shows a summary of the layer 
  //tfvis.show.layer(surface, model.getLayer(undefined, 1));

  //Get weights of the layer, returns tensor - this is kernal
  let weights = layer.getWeights()[0].dataSync();
  let kernalTensor = layer.getWeights()[0];
  console.log(kernalTensor.shape)

  //Getting weights via model
  let weightsM = model.getWeights()[0].dataSync();
  //console.log(weightsM)

  

  //Find max and min 
  let maxW = d3.max(weights)
  let minW = d3.min(weights)
  console.log(maxW,minW)

  //Map weight to  scale from 0 to 255
  let imScale = d3.scaleLinear().domain([minW,maxW]).range([0,255]);
  let weightsSc = weights.map(m=> parseInt(imScale(m)));
  //console.log(weightsSc)

  //Change to regular array object: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Typed_arrays
  let scaledArr = Array.from(weightsSc);
  //console.log(scaledArr)

  //Convert to Uint8ClampedArray
  var width = 28,
    height = 28,
    buffer = new Uint8ClampedArray(width * height * 4); // have enough bytes

  let i = 784*0;
  for(var y = 0; y < height; y++) {
    for(var x = 0; x < width; x++) {
        var pos = (y * width + x) * 4; // position in buffer based on x and y
        buffer[pos  ] = scaledArr[i]-50;           // some R value [0, 255]
        buffer[pos+1] = scaledArr[i];           // some G value
        buffer[pos+2] = scaledArr[i];           // some B value
        buffer[pos+3] = 255;          // set alpha channel
        i++;
    }
  }
  

  // // kernel:
  // model.layers[0].getWeights()[0].print()
  // // bias:
  // model.layers[0].getWeights()[1].print()

  //console.log("layer",layer)
  //console.log("weights",weights)
  
  //Number of nodes 
  let num_nodes = scaledArr.length/784;
  let batchSize = num_nodes;
  let IMAGE_SIZE = 784;

  //Drawing w/out converting to tensor
  //This convertes to array of length 10 composed of image data
  let weightMat = [];
  while(scaledArr.length) weightMat.push(scaledArr.splice(0,IMAGE_SIZE));
  //console.log(weightMat)


  const div = d3.select(`#layer`).append("div")
    .attr("class","weight-container");

  const canvas = div.append('canvas')
    .attr("class","weight")
    .attr("width",28)
    .attr("height",28)
    .style("margin","4px")
    .node();
  var ctx = canvas.getContext('2d');
  var ImageData = ctx.createImageData(28, 28);

  
  // Sets buffer as source
  ImageData.data.set(buffer)

  // update canvas with new data
  ctx.putImageData(ImageData, 0, 0);



  //Convert weights to tensor
  // let weights_tensor = tf.tensor2d(weightsSc, [10, IMAGE_SIZE]);
  // const weightsMax = weights_tensor.max();
  // const weightsMin = weights_tensor.min();  
  // // console.log(weightsMax,weightsMin)

  // weights_tensor = weights_tensor.sub(weightsMin).div(weightsMax.sub(weightsMin));
  kernalTensor = kernalTensor.sub(kernalTensor.min()).div(kernalTensor.max().sub(kernalTensor.min()));
  // // console.log(weights_tensor)


  // //Now, I want to package these into 28x28 images and display them, similar to what
  // // //we did in "showexamples"
  for (let i = 0; i < num_nodes; i++) {

    let imageTensor = tf.tidy(() => {          //Tidy helps to prevent memory leakage
      // Reshape the image to 28x28 px
      return kernalTensor
        //2-D tensor, specifies slicing from row, and taking out size of image: https://www.quora.com/How-does-tf-slice-work-in-TensorFlow
        .slice([0, i], [kernalTensor.shape[0],1])
        // //Reshapes to image size 
        .reshape([28, 28, 1]);
    });



  //Using D3 because...well...I like it.
  const div = d3.select(`#layer`).append("div")
  .attr("class","weight-container");

  //Create canvas element
  const canvas = div.append('canvas')
    .attr("class","weight")
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

  kernalTensor.dispose()


}





async function showAccuracy(model, data) {
  let [preds, labels] = doPrediction(model, data,500);
  preds = preds.argMax([-1]);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  let [preds, labels] = doPrediction(model, data,500);
  preds = preds.argMax([-1]);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames);

  labels.dispose();
}



// async function writeInternalActivationAndGetOutput(
//   model, layerNames, inputImage, numFilters, outputDir) {
// const layerName2FilePaths = {};
// const layerOutputs =
//     layerNames.map(layerName => model.getLayer(layerName).output);
// const compositeModel = tf.model(
//     {inputs: model.input, outputs: layerOutputs.concat(model.outputs[0])});

// const outputs = compositeModel.predict(inputImage);

// for (let i = 0; i < outputs.length - 1; ++i) {
//   const layerName = layerNames[i];
//   const activationTensors =
//       tf.split(outputs[i], outputs[i].shape[outputs[i].shape.length - 1], -1);
//   const actualNumFilters = filters <= activationTensors.length ?
//       numFilters :
//       activationTensors.length;
//   const filePaths = [];
//   for (let j = 0; j < actualNumFilters; ++j) {
//     const imageTensor = tf.tidy(
//         () => deprocessImage(tf.tile(activationTensors[j], [1, 1, 1, 3])));
//     const outputFilePath = path.join(outputDir, `${layerName}_${j + 1}.png`);
//     filePaths.push(outputFilePath);
//     await utils.writeImageTensorToFile(imageTensor, outputFilePath);
//   }
//   layerName2FilePaths[layerName] = filePaths;
//   tf.dispose(activationTensors);
// }
// tf.dispose(outputs.slice(0, outputs.length - 1));
// return {modelOutput: outputs[outputs.length - 1], layerName2FilePaths};
// }
