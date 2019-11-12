/** Adapted from: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?index=..%2F..index#2 */
import {MnistData} from './data.js';



async function showExamples(data) {
  // Get the examples
  const examples = data.nextTestBatch(13); //selects images
  const numExamples = examples.xs.shape[0]; //returns number of examples by capturing the "rows" of the tensor
  const labels = Array.from(examples.labels.argMax(1).dataSync()); //Returns list of labels 
  //console.log(numExamples)
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
    const div = d3.select("#load-view").append("div")
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
  await showExamples(data);

  const model = getModel();
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  
  await train(model, data);
}

document.addEventListener('DOMContentLoaded', run);




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

    //Creates a dense (fully connected) layer.
        // This layer implements the operation: output = activation(dot(input, kernel) + bias)
        // activation is the element-wise activation function passed as the activation argument.
        // kernel is a weights matrix created by the layer.
        // bias is a bias vector created by the layer (only applicable if useBias is true).
    //Allow user selected different activation functions here + change number of layers
    model.add( tf.layers.dense( { units: 50, activation: 'relu' } ) )
    model.add( tf.layers.dense( { units: 50, activation: 'relu' } ) ) 

    //Softmax layer at end
    model.add( tf.layers.dense( { units: 10, activation: 'softmax' } ) ) 




    /** LINEAR MODEL */




    
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
      metrics: ['accuracy'], //Might be able to add precision or recall here too?
    });
  
    return model;
  }




  /** Training with the model */

  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];

    //Visualizing these training metrics
    const container = {
      name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    //Allow user to define these
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    //Sets up training tensors
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    //Sets up test tensors
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    // model.fit starts the training loop
    let NUM_EPOCHS = 10;
    
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: NUM_EPOCHS,
      shuffle: true,
      callbacks: 
          fitCallbacks

      
    });
  }
