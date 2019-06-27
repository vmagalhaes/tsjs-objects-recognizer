import React from 'react';
import './App.css';

import * as cocoSsd from '@tensorflow-models/coco-ssd';

import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

class CocoObjectDetection extends React.Component {

  constructor(props) {
    super(props);
    // this.imageRef = React.createRef();
    this.videoRef = React.createRef();
    // this.imageCanvasRef = React.createRef();
    this.videoCanvasRef = React.createRef();

    this.state = {
      predictions: [],
      imageStatus: "loading"
    }
  }

  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = cocoSsd.load();
      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model) => {
    model.detect(video).then(predictions => {
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
    });
  };

  renderPredictions = predictions => {
    const ctx = this.videoCanvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(prediction.class, x, y);
    });
  };

  // cropToCanvas = () => {
  //   const image = this.imageRef.current;
  //   const canvas = this.imageCanvasRef.current;
  //   const ctx = canvas.getContext("2d");
  //   const naturalWidth = image.naturalWidth;
  //   const naturalHeight = image.naturalHeight;

  //   canvas.width = image.width;
  //   canvas.height = image.height;

  //   ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  //   if (naturalWidth > naturalHeight) {
  //     ctx.drawImage(
  //       image,
  //       (naturalWidth - naturalHeight) / 2,
  //       0,
  //       naturalHeight,
  //       naturalHeight,
  //       0,
  //       0,
  //       ctx.canvas.width,
  //       ctx.canvas.height
  //     );
  //   } else {
  //     ctx.drawImage(
  //       image,
  //       0,
  //       (naturalHeight - naturalWidth) / 2,
  //       naturalWidth,
  //       naturalWidth,
  //       0,
  //       0,
  //       ctx.canvas.width,
  //       ctx.canvas.height
  //     );
  //   }
  // };

  // onImageLoad() {
  //   cocoSsd.load()
  //     .then(model => model.detect(this.imageRef.current))
  //     .then(predictions => {
  //       console.log(predictions);
  //       this.cropToCanvas();

  //       const font = "16px sans-serif";
  //       const ctx = this.imageCanvasRef.current.getContext('2d');
  //       ctx.font = font;
  //       ctx.textBaseline = "top";

  //       predictions.forEach(prediction => {
  //         const x = prediction.bbox[0];
  //         const y = prediction.bbox[1];
  //         const width = prediction.bbox[2];
  //         const height = prediction.bbox[3];
  //         // Draw the bounding box.
  //         ctx.strokeStyle = "#00FFFF";
  //         ctx.lineWidth = 4;
  //         ctx.strokeRect(x, y, width, height);
  //         // Draw the label background.
  //         ctx.fillStyle = "#00FFFF";
  //         const textWidth = ctx.measureText(prediction.class).width;
  //         const textHeight = parseInt(font, 10); // base 10
  //         ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
  //       });

  //       predictions.forEach(prediction => {
  //         const x = prediction.bbox[0];
  //         const y = prediction.bbox[1];
  //         ctx.fillStyle = "#000000";
  //         ctx.fillText(prediction.class, x, y);
  //       });

  //       this.setState({ predictions });
  //     });
  // }

  render() {
    return (
      <div className="App">
        {/* <img
          alt="Cat"
          ref={this.imageRef}
          onLoad={this.onImageLoad(this)}
          id="image"
          width="500px"
          crossOrigin="anonymous"
          src="https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg"
        /> */}
        <video
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="600"
          height="500"
        />
        {/* <canvas
          className="image-canvas"
          ref={this.imageCanvasRef}
          width="500"
          height="500"
        /> */}
        <canvas
          className="size"
          ref={this.videoCanvasRef}
          width="600"
          height="500"
        />
      </div>
    );
  }
}

class KnnClassifierComponent extends React.Component {

  constructor(props) {
    super(props);

    this.videoCanvasRef = React.createRef();

    this.state = {
      video: undefined,
      videoWidth: 300,
      videoHeight: 250,
      NUM_CLASSES: 3,
      TOPK: 3,
      infoTexts: [],
      training: -1,
      modelPromise: undefined,
      classifier: undefined
    }
  }

  async animate() {
    // Get image data from video element
    const video = this.state.video;
    const image = tf.browser.fromPixels(video);
    let logits;
    // 'conv_preds' is the logits activation of MobileNet.
    const infer = () => this.state.modelPromise.infer(image, 'conv_preds');

    // Train class if one of the buttons is held down
    if (this.state.training !== -1) {
      logits = infer();
      // Add current image to classifier
      this.state.classifier.addExample(logits, this.state.training);
    }

    // If the classifier has examples for any classes, make a prediction!
    const numClasses = await this.state.classifier.getNumClasses();
    if (numClasses > 0) {
      logits = infer();

      const res = await this.state.classifier.predictClass(logits, this.state.TOPK);
      for (let i = 0; i < this.state.NUM_CLASSES; i++) {
        // Make the predicted class bold
        if (res.classIndex === i) {
          this.state.infoTexts[i].style.fontWeight = 'bold';
        } else {
          this.state.infoTexts[i].style.fontWeight = 'normal';
        }

        const classExampleCount = this.state.classifier.getClassExampleCount();
        // Update info text
        if (classExampleCount[i] > 0) {
          const conf = res.confidences[i] * 100;
          this.state.infoTexts[i].innerText = ` ${classExampleCount[i]} examples - ${conf}%`;

          const innerHTML = this.state.infoTexts.map((element, index) => {
            const value = parseInt(element.innerText.slice(element.innerHTML.length - 4, element.innerHTML.length - 1));

            return {
              index,
              value: value ? value : 0  }
            }
          );

          const highest = innerHTML.sort((a, b) => b.value - a.value)[0]
          switch (highest.index) {
            case 0:
              document.body.style.background = "#288340";
              break;
            case 1:
              document.body.style.background = "#9B1617";
              break;
            case 2:
              document.body.style.background = "#083C84";
              break;
          }
        }
      }
    }

    image.dispose();
    if (logits != null) {
      logits.dispose();
    }

    requestAnimationFrame(this.animate.bind(this));
  }

  componentDidMount() {
    document.onreadystatechange = () => {
      if (document.readyState === 'complete') {
        this.setupGui()
        this.setupCam()
      }
    };
  }

  setupGui() {
    // Create training buttons and info texts
    for (let i = 0; i < this.state.NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button');
      button.innerText = 'Treinar ' + i;
      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('click', () => {
        this.setState(previousState => ({
          ...previousState,
          training: i
        }));

        this.animate();

        requestAnimationFrame(() => {
          this.setState(previousState => ({
            ...previousState,
            training: -1
          }));
        });
      });

      // Create info text
      const infoText = document.createElement('span');
      infoText.innerText = ' Nenhum exemplo criado';
      div.appendChild(infoText);
      this.state.infoTexts.push(infoText);
    }
  }

  async setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
          'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    this.state.video = document.getElementById('video');
    this.state.video.width = this.state.videoWidth;
    this.state.video.height = this.state.videoHeight;

    const stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': {
        facingMode: 'user',
        width: this.state.videoWidth,
        height: this.state.videoHeight,
      },
    });
    this.state.video.srcObject = stream;

    return new Promise((resolve) => {
      this.state.video.onloadedmetadata = () => {
        resolve(this.state.video);
      };
    });
  }

  async setupCam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = await this.setupCamera();
      const modelPromise = cocoSsd.load();

      this.setState(previousState => ({
        ...previousState,
        modelPromise: mobilenetModule.load(2),
        classifier: knnClassifier.create()
      }));

      Promise.all([this.state.modelPromise, webCamPromise, modelPromise])
        .then(values => {
          this.setState(previousState => ({
            ...previousState,
            modelPromise: values[0]
          }));

          this.detectFrame(values[1], values[2]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model) => {
    model.detect(video).then(predictions => {
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
    });
  };

  renderPredictions = predictions => {
    const ctx = this.videoCanvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(prediction.class, x, y);
    });
  };

  render() {
    return (
      <div>
        <video
          id="video"
          autoPlay
          playsInline
          muted
        />
        <canvas
          className="size"
          ref={this.videoCanvasRef}
          width="300"
          height="250"
        />
      </div>
    );
  }
}


class TensorFlowComponents extends React.Component {
  render() {
    return (
      <div>
        {/* <CocoObjectDetection /> */}
        <KnnClassifierComponent />
      </div>
    )
  }
}

export default TensorFlowComponents;