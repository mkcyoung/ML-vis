


// let surface = d3.select("#layer").node();

  // .output does something which might be useful
  // let layerOutput = layer.output
  //console.log(layerOutput)

  //Shows a summary of the layer 
  //tfvis.show.layer(surface, model.getLayer(undefined, 1));

  //Get weights of the layer, returns tensor - this is kernal
  // let weights = layer.getWeights()[0].dataSync();


  
//Getting weights via model
  // let weightsM = model.getWeights()[0].dataSync();
  //console.log(weightsM)

  

  //Find max and min 
  // let maxW = d3.max(weights)
  // let minW = d3.min(weights)
  // console.log(maxW,minW)

  //Map weight to  scale from 0 to 255
  // let imScale = d3.scaleLinear().domain([minW,maxW]).range([0,255]);
  // let weightsSc = weights.map(m=> parseInt(imScale(m)));
  //console.log(weightsSc)

  //Change to regular array object: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Typed_arrays
  // let scaledArr = Array.from(weightsSc);
  //console.log(scaledArr)

  // //Convert to Uint8ClampedArray
  // var width = 28,
  //   height = 28,
  //   buffer = new Uint8ClampedArray(width * height * 4); // have enough bytes

  // let i = 784*0;
  // for(var y = 0; y < height; y++) {
  //   for(var x = 0; x < width; x++) {
  //       var pos = (y * width + x) * 4; // position in buffer based on x and y
  //       buffer[pos  ] = scaledArr[i]-50;           // some R value [0, 255]
  //       buffer[pos+1] = scaledArr[i];           // some G value
  //       buffer[pos+2] = scaledArr[i];           // some B value
  //       buffer[pos+3] = 255;          // set alpha channel
  //       i++;
  //   }
  // }
  

  // // kernel:
  // model.layers[0].getWeights()[0].print()
  // // bias:
  // model.layers[0].getWeights()[1].print()

  //console.log("layer",layer)
  //console.log("weights",weights)


// //Drawing w/out converting to tensor
  // //This convertes to array of length 10 composed of image data
  // let weightMat = [];
  // while(scaledArr.length) weightMat.push(scaledArr.splice(0,IMAGE_SIZE));
  // //console.log(weightMat)


  // const div = d3.select(`#layer`).append("div")
  //   .attr("class","weight-container");

  // const canvas = div.append('canvas')
  //   .attr("class","weight")
  //   .attr("width",28)
  //   .attr("height",28)
  //   .style("margin","4px")
  //   .node();
  // var ctx = canvas.getContext('2d');
  // var ImageData = ctx.createImageData(28, 28);

  
  // // Sets buffer as source
  // ImageData.data.set(buffer)

  // // update canvas with new data
  // ctx.putImageData(ImageData, 0, 0);