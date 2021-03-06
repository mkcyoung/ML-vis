/** Drawing Functions 
 * From: http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/
*/

export class Drawing {
    constructor() {
      this.model = null;
      this.modelAct = null;

    }

    drawing(){

        let that = this;
        //console.log("in drawing: ",this.model)
        //console.log("in drawing",this.modelAct)
        //Defines our context
        this.context = d3.select('#drawn-digit').node().getContext("2d");

        
        //Mousedown
        $('#drawn-digit').mousedown(function(e){
        var mouseX = e.pageX - this.offsetLeft;
        var mouseY = e.pageY - this.offsetTop;
                
        paint = true;
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
        redraw();
        });

        //Mousemove
        $('#drawn-digit').mousemove(function(e){
        if(paint){
            addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
            redraw();
        }
        });

        //Mouseup
        $('#drawn-digit').mouseup(function(e){
        paint = false;
        });

        //mouseleave
        $('#drawn-digit').mouseleave(function(e){
        paint = false;
        });


        //Add click
        this.clickX = new Array();
        this.clickY = new Array();
        this.clickDrag = new Array();
        var paint;

        function addClick(x, y, dragging)
        {
        that.clickX.push(x);
        that.clickY.push(y);
        that.clickDrag.push(dragging);
        }


        //Redraw - where the magic happens
        function redraw(){
            that.context.clearRect(0, 0, that.context.canvas.width, that.context.canvas.height); // Clears the canvas
            
            that.context.strokeStyle = "white";
            that.context.lineJoin = "round";
            that.context.lineWidth = 20;
                    
            for(var i=0; i < that.clickX.length; i++) {		
            that.context.beginPath();
            if(that.clickDrag[i] && i){
                that.context.moveTo(that.clickX[i-1], that.clickY[i-1]);
            }else{
                that.context.moveTo(that.clickX[i]-1, that.clickY[i]);
            }
            that.context.lineTo(that.clickX[i], that.clickY[i]);
            that.context.closePath();
            that.context.stroke();
            }

            //Shows the prediction as the image is being drawn
            showPrediction(that.model)

            //Shows activation maps as the image is being drawn
            showActivation(that.modelAct)

        }

        //Clear button
        d3.select("#clear-button")
            .on("click",function(){
                // Clears the drawing canvas
                that.context.clearRect(0, 0, that.context.canvas.width, that.context.canvas.height); 
                //Clears resized digit canvas
                d3.select("#resized-digit").node().getContext("2d").clearRect(0, 0, 28, 28) 
                //Erases arrays with stored info
                that.clickX = [];
                that.clickY = [];
                that.clickDrag = [];

                //Updates predictions histo
                showPrediction(that.model)
                
            });


        //Evaluating the model / making predictions
        const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

        //Makes predictions once model is trained - new data there aren't existing labels
        function doPrediction(model) {
            //doPrediction(model, data, testDataSize)
            const IMAGE_WIDTH = 28;
            const IMAGE_HEIGHT = 28;
            //This selects a random examples from the data
            // const testData = data.nextTestBatch(testDataSize);
            // const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
            // const labels = testData.labels.argMax([-1]);
            // const preds = model.predict(testxs);//.argMax([-1]);
        
            //This selects drawn canvas image
            let drawnCanvas = $('#drawn-digit')[0];
        
            //Need to resize the image to 28x28
            let resizedCanvas = d3.select("#resized-digit").node()
            let context = resizedCanvas.getContext("2d");
            context.drawImage(drawnCanvas, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
            const imageData = context.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
        
            //Creates input tensor to pass into prediction
            const inputTensor = tf.browser.fromPixels(imageData, 1)
                .reshape([1, 28, 28, 1])
                .cast('float32')
                .div(255);
            
            //console.log(inputTensor.shape)
            
            const preds = model.predict(inputTensor);
        
        
            //This shows me the output of softmax along with actual label
            //console.log(model.predict(inputTensor).dataSync(),testData.labels.argMax([-1]).dataSync())
        
            //testxs.dispose();
            inputTensor.dispose();
        
            return [preds] //, labels, testData];
        }
        
        
        
        /**
         * This function shows a prediction example and its corrresponding softmax histogram
         * @param {the training model} model 
         * @param {the input data} data 
         */
        async function showPrediction(model){
            //Generates 
            let [preds] = doPrediction(model);
            //let [preds,labels,testData] = doPrediction(model, data, 1);
        
            //Makes a barchart showing a prediction for a single example
            let myBarChart = d3.select("#histo").node();
            let barchartData = Array.from(preds.dataSync()).map((d, i) => {
            return { index: i, value: d }
            })
            tfvis.render.barchart(myBarChart, barchartData,  { width: 500, height: 200, fontSize:18 })
        
            //Shows the predicted example
            //showExamples(testData,"input",1)

            //Renders text showing model's prediciton
            let predText = d3.select("#pred-text")
                .text(preds.argMax([-1]).dataSync())
        
        }


        /**
         * This function shows the activation maps corresponding to the layer filters as 
         * the image is being drawn
         * @param {the training model} model 
         * @param {the input data} data 
         */
        async function showActivation(model){

            //console.log("in showActivation")


            //Retrieves output, which is the set of activation maps
            let [actMaps] = doPrediction(model);
            //console.log("actMap shape:",actMaps.shape)

            //normalizes tensor weights so they can be mapped to pixels by tf.browser.toPixels
            actMaps = actMaps.sub(actMaps.min()).div(actMaps.max().sub(actMaps.min()));

            let imageTensor = null;

            for (let i = 0; i < 8; i++) {

                imageTensor = tf.tidy(() => {          //Tidy helps to prevent memory leakage
                    // Reshape the image to 28x28 px
                    return actMaps
                      //4-D tensor, specifies slicing from row, and taking out size of image: https://www.quora.com/How-does-tf-slice-work-in-TensorFlow
                      .slice([0, 0, 0, i], [1,24,24,1]) // slices columns up to size of images
                      .reshape([24, 24, 1]);
                    
                  });

                //console.log("shape after processing:",imageTensor.shape)
                //Convert tensors to canvas images
                await tf.browser.toPixels(imageTensor, d3.select(`#act_${i}`).node()); //Draws tensor of pixel values to byte array or canvas in this case

                //Draw canvases to div -> maybe use enter-exit here... this doesn't get rid of old divs.
                d3.select(`#act-container-${i}`).node().appendChild(d3.select(`#act_${i}`).node());

                //Cleans up tensor, again a memory thing.
                imageTensor.dispose(); 

            }
            actMaps.dispose();

        
        }

    }

}