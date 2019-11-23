/** Drawing Functions 
 * From: http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/
*/

export class Drawing {
    constructor() {
      this.model = null;

    }

    drawing(){

        let that = this;
        //console.log("in drawing: ",this.model)
        //Defines our context
        let context = d3.select('#drawn-digit').node().getContext("2d");

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
        var clickX = new Array();
        var clickY = new Array();
        var clickDrag = new Array();
        var paint;

        function addClick(x, y, dragging)
        {
        clickX.push(x);
        clickY.push(y);
        clickDrag.push(dragging);
        }


        //Redraw - where the magic happens
        function redraw(){
            context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
            
            context.strokeStyle = "white";
            context.lineJoin = "round";
            context.lineWidth = 20;
                    
            for(var i=0; i < clickX.length; i++) {		
            context.beginPath();
            if(clickDrag[i] && i){
                context.moveTo(clickX[i-1], clickY[i-1]);
            }else{
                context.moveTo(clickX[i]-1, clickY[i]);
            }
            context.lineTo(clickX[i], clickY[i]);
            context.closePath();
            context.stroke();
            }

            showPrediction(that.model)

        }

        //Will add button to clear at some point
        //   if(button pressed){
        //     context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
            
        //   }



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
            //console.log(model.predict(testxs).dataSync(),testData.labels.argMax([-1]).dataSync())
        
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
        }

    }

}