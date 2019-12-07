/** Adapted from: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?index=..%2F..index#2 */

import {MnistData} from './data.js';
import {Drawing} from './drawing.js';


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
        .text(labels[i]);

    //Create canvas element
    const canvas = div.append('canvas')
        .attr("class","pre-view")
        .attr("width",28)
        .attr("height",28)
        .style("margin","1px")
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
  await showExamples(data,"load-view",12);

  let that = this;

  //Sets up network type variable
  this.convo = false;

  d3.select("#basic-button")
    .on("click", function(){
        that.convo = false;
        d3.select("#layer-num")
          .style("visibility","visible");
        d3.select(".label-layer")
          .style("visibility","visible");
      });

  d3.select("#convolution-button")
    .on("click", function(){
        that.convo = true;
        d3.select("#layer-num")
          .style("visibility","hidden");
        d3.select(".label-layer")
          .style("visibility","hidden");
      });

  //Tooltips for network types
  //make tooltip div for descriptions
  d3.select(".wrapper")
    .append("div")
    .attr("id", "basic-tooltip")
    .style("opacity", 0);

  //initiate on mouseover events
  let basic_description = d3.select("#basic-button");
  let convo_description = d3.select("#convolution-button");

  basic_description
    .on("mouseover",function(){
      d3.select("#basic-tooltip")
          .transition()
          .duration(200)
          .style("opacity", 1);
      d3.select("#basic-tooltip").html("<p> A traditional dense (fully connected) network. Each layer has 10 units. We input our 28x28 pixel images as unrolled 784 element feature vectors which feed into every node of the first layer. Scroll down after hitting the train button to see how these first layer nodes look during training. </p>");
          // .style("left",(d3.event.pageX+15) + "px") 
          // .style("top", (d3.event.pageY+15) + "px");     
    })
    .on("mouseout",function(){
      d3.select("#basic-tooltip")
          .transition()
          .duration(200)
          .style("opacity", 0);
     });

  convo_description
    .on("mouseover",function(){
      d3.select("#basic-tooltip")
          .transition()
          .duration(200)
          .style("opacity", 1);
      d3.select("#basic-tooltip").html("<p> A convolutional neural network (CNN). CNN's excel at image recognition tasks by building up complex feature detection from simpler features in an efficient way. They accomplish this by introducing convolutional filters which detect the presence of certain features in input images. These filters convolve with the input to produce activation maps which show the presence of that feature across the input. Scroll down after training to explore these elements of the CNN.  </p>");
          // .style("left",(d3.event.pageX+15) + "px") 
          // .style("top", (d3.event.pageY+15) + "px");     
    })
    .on("mouseout",function(){
      d3.select("#basic-tooltip")
          .transition()
          .duration(200)
          .style("opacity", 0);
    });


  //Initializes the model with the selected params
  let model = null;
  let layers = null;
  let modelAct = null;
  d3.select("#init-button")
    .on("click",function(){
        model=null;
        model = getModel(that.convo);

         // get all the layers of the model
        layers = model.layers

        // second model, used for visualizing activations
        modelAct = tf.model({
          inputs: layers[0].input, 
          outputs: layers[0].output
        })

        //Visualizing model summary
        let container = d3.select("#model-summary");
        //tfvis.show.modelSummary(container.node(), model);
        tfvis.show.modelSummary(container.node(),model);
        tfvis.show.layer(container.node(), model);
        //tfvis.show.modelSummary({name: 'Model Architecture'}, model);

        //Selects and removes all previous model weight visualizations
        d3.selectAll(".weight-container").remove();
        d3.selectAll(".act-container").remove();
    });

 
  
  const drawing = new Drawing();
  //Select train button and train on click
  d3.select("#train-button")
    .on("click", function(){
      //Pass model into drawing.js
      drawing.model = model;
      //Passes activation model into drawing
      drawing.modelAct = modelAct;
      drawing.drawing();

      return train(model, data, that.convo);
      //previously await train(model, data)
    });
    

  //Prediction and evaluation
  //await showAccuracy(model, data);
  //await showConfusion(model, data);

  

  //await showPrediction(model);

}

document.addEventListener('DOMContentLoaded', run);



/** Model architecture 
 * I want to eventually have some controls that adjust parameters within this.
*/

function getModel(convo) {
    console.log(convo)
    let model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    let num_layers = parseInt(d3.select("#layer-num").node().value);
    console.log("layer num:",num_layers);

    //Select learning rate from form
    let learning_rate = d3.select("#learn-rate").node().value;
    console.log("learning rate: ",learning_rate)
    //let learning_rate = 0.05;

    

    //If basic selected, model returns dense network
    if (convo != true){

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
      let layerString = "model.add( tf.layers.dense( { units: 10, activation: 'sigmoid' } ) ); ";
      let layerStringMult = layerString.repeat(num_layers-1);
      console.log(layerStringMult)
      eval(layerStringMult);
      


      //model.add( tf.layers.dense( { units: 20, activation: 'relu' } ) )

      //Softmax layer at end
      model.add( tf.layers.dense( { units: 10, activation: 'softmax'} ) ) //'softmax

      // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    // Constructs a tf.AdamOptimizer that uses the Adam algorithm. See https://arxiv.org/abs/1412.6980

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

    }

    else{
      /** Convolutional Neural Network */

      // In the first layer of our convolutional neural network we have 
      // to specify the input shape. Then we specify some parameters for 
      // the convolution operation that takes place in this layer.
      model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));

      // // The MaxPooling layer acts as a sort of downsampling using max values
      // // in a region instead of averaging.  
      model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
      
      // // Repeat another conv2d + maxPooling stack. 
      // // Note that we have more filters in the convolution.
      model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));
      model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
      
      // // Now we flatten the output from the 2D filters into a 1D vector to prepare
      // // it for input into our last layer. This is common practice when feeding
      // // higher dimensional data to a final classification output layer.
      model.add(tf.layers.flatten());
    
      // // Our last layer is a dense layer which has 10 output units, one for each
      // // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
      const NUM_OUTPUT_CLASSES = 10;
      model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
      }));
    
      
      // Choose an optimizer, loss function and accuracy metric,
      // then compile and return the model
      // Constructs a tf.AdamOptimizer that uses the Adam algorithm. See https://arxiv.org/abs/1412.6980

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
  }
  
    return model;
  }




  /** Training with the model */

  async function train(model, data, convo) {
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
    const BATCH_SIZE = parseInt(d3.select("#batch").node().value);
    console.log("batch",BATCH_SIZE)
    
    const TRAIN_DATA_SIZE = parseInt(d3.select("#training-num").node().value);
    console.log("training size",TRAIN_DATA_SIZE)
    const TEST_DATA_SIZE = parseInt(d3.select("#test-num").node().value);
    console.log("testing size",TEST_DATA_SIZE)
  
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
    let num_units=null;
    (convo) ?  num_units = 8 : num_units = 10; //number of nodes to visualize in a layer
     

    for (let i = 0; i < num_units; i++){
      let div = d3.select(`#layer`).append("div")
        .attr("class",`weight-container`)
        .attr("id",`weight-container-${i}`);

      let canvas = div.append('canvas')
        .attr("id",`weight_${i}`)
        .attr("class","weight")
        .attr("width",28)
        .attr("height",28)
        .style("margin","4px")
        .node();
    }

    if (convo){
      //console.log("setting up convo divs")

      //Setting up activation map divs to be used if convo is selected
      for (let i = 0; i < num_units; i++){
        let div = d3.select(`#actMaps`).append("div")
          .attr("class",`act-container`)
          .attr("id",`act-container-${i}`);

        let canvas = div.append('canvas')
          .attr("id",`act_${i}`)
          .attr("class","act")
          .attr("width",24)
          .attr("height",24)
          .style("margin","4px")
          .node();
      }
    }

    // model.fit starts the training loop
    let NUM_EPOCHS = parseInt(d3.select("#epoch").node().value);
    console.log("epochs",NUM_EPOCHS)


    const trainLogs = [];
    const lossContainer = document.getElementById('training-view');
    const accContainer = document.getElementById('acc-view')
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
            trainLogs.push(logs);
            tfvis.show.history(lossContainer, trainLogs, metrics, { width: 590, height: 200 })
            //const [{xs: xTest, ys: yTest}] = await validationData.toArray();

            showConfusion(model,testXs,testYs)
              
           
            //Calls the showLayer function to show layer weights at end of every epoch
            await showLayer(model,convo)
            
            
          },
          //Visualized nodes after batch processed, gives smoother image but slows training
          // onBatchEnd: async (batch,logs) => {
          //     //await showLayer(model)
          //     //Could also potentially store weights for each node here for later perusal using sliderbar
          // }
          
        }
    });
    return model


  }


/**
 * This function visualizes nodes in layers
 * @param {layer data} data 
 */
async function showLayer(model,convo,ctx,ImageData){
  //console.log("in showlayer: ",convo)

  //First step is to retrieve all of the weights from the designated layer
  //Can select by name: ('dense_Dense1' ) or by position: (undefined,1)
  // let layer = model.getLayer(undefined,1)

  // //for convo
  // let layer = model.getLayer(undefined,0)

  let layer = null;

  (convo) ? layer = model.getLayer(undefined,0) : layer = model.getLayer(undefined,1)

  //Gets kernal (weights) from selected layer ([0] = kernal, [1] = bias)
  //For convo - getWeights()[0] = shape=> 5 5 1 8
  // think the 5x5 are the filters , then there are 8 of them
  //Can visualize these filter then also visualization "activation maps"
  // getWeights()[1] = shape=> 16
  let kernalTensor = layer.getWeights()[0];
  console.log("shape of weights: ",kernalTensor.shape)

  let num_nodes = null;
  if (convo != true){
    //Number of nodes, clamped at 10 (will change once I have good way to visualize)
    num_nodes = (kernalTensor.shape[1] > 10 ) ? 10 : kernalTensor.shape[1]; //# columns of kernal tensor = output nodes
  }
  else{
    //for conv
    num_nodes = (kernalTensor.shape[1] > 10 ) ? 10 : kernalTensor.shape[3]; 
  }

  //normalizes tensor weights so they can be mapped to pixels by tf.browser.toPixels
  kernalTensor = kernalTensor.sub(kernalTensor.min()).div(kernalTensor.max().sub(kernalTensor.min()));

  //Now, I want to package these into 28x28 images and display them, similar to what
  //we did in "showexamples"
  let imageTensor = null;

  for (let i = 0; i < num_nodes; i++) {

    if(convo != true){
      imageTensor = tf.tidy(() => {          //Tidy helps to prevent memory leakage
        // Reshape the image to 28x28 px
        return kernalTensor
          //2-D tensor, specifies slicing from row, and taking out size of image: https://www.quora.com/How-does-tf-slice-work-in-TensorFlow
          .slice([0, i], [kernalTensor.shape[0],1]) // slices columns up to size of images
          // //Reshapes to image size 
          .reshape([28, 28, 1]);
      });

  }
  else{
      //For conv
      imageTensor = tf.tidy(() => {          //Tidy helps to prevent memory leakage
        // Reshape the image to 28x28 px
        return kernalTensor
          //4-D tensor, specifies slicing from row, and taking out size of image: https://www.quora.com/How-does-tf-slice-work-in-TensorFlow
          .slice([0, 0, 0, i], [5,5,1,1]) // slices columns up to size of images
          .reshape([5, 5, 1]);
        
      });
  }

    //console.log("shape after processing:",imageTensor.shape)
    //Convert tensors to canvas images
    await tf.browser.toPixels(imageTensor, d3.select(`#weight_${i}`).node()); //Draws tensor of pixel values to byte array or canvas in this case

    //Draw canvases to div -> maybe use enter-exit here... this doesn't get rid of old divs.
    d3.select(`#weight-container-${i}`).node().appendChild(d3.select(`#weight_${i}`).node());

    //Cleans up tensor, again a memory thing.
    imageTensor.dispose(); 

  }
  kernalTensor.dispose()

}

//Evaluating the model / making predictions
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

// function doPrediction(model, data, testDataSize = 500) {
//   const IMAGE_WIDTH = 28;
//   const IMAGE_HEIGHT = 28;
//   const testData = data.nextTestBatch(testDataSize);
//   const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
//   const labels = testData.labels.argMax([-1]);
//   const preds = model.predict(testxs).argMax([-1]);

//   testxs.dispose();
//   return [preds, labels];
// }

//Shows accuracy list in visor tab
// async function showAccuracy(model, data, container) {
//   let [preds, labels] = doPrediction(model, data,500);
//   preds = preds.argMax([-1]);
//   const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
//   const container = {name: 'Accuracy', tab: 'Evaluation'};
//   tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

//   labels.dispose();
// }

//Shows confusion matrix in visor tab
async function showConfusion(model, xTest, yTest) {

  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });


  const confMatrixData = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('acc-container');
  tfvis.render.confusionMatrix(
      container, {values: confMatrixData, labels: classNames},
      {shadeDiagonal: true,
      },
  );

  tf.dispose([preds, labels]);

  // let [preds, labels] = doPrediction(model, data);
  // preds = preds.argMax([-1]);
  // const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  // const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  // tfvis.render.confusionMatrix(
  //     container, {values: confusionMatrix}, classNames);

  // labels.dispose();

}


