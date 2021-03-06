import React from "react";
import "./App.css";
import * as ml5 from 'ml5'
import Sketch from "react-p5";
import classification_model from './model3/model.json'
import classification_model_meta from './model3/model_meta.json'

function App() {

  let video;
  let poseNet;
  let pose;
  let skeleton;

  let brain;
  let poseLabel = "Y";

  const setup = (p5, canvasParentRef) => {
    p5.createCanvas(640, 480).parent(canvasParentRef);
    video = p5.createCapture(p5.VIDEO);
    video.hide();
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on('pose', gotPoses);
  
    let options = {
      inputs: 34,
      outputs: 5,
      task: 'classification',
      debug: true
    }
    brain = ml5.neuralNetwork(options);
    const modelInfo = {
      model: 'model3/model.json',
      metadata: 'model3/model_meta.json',
      weights: 'model3/model.weights.bin',
    };
    brain.load(modelInfo, brainLoaded);
  }
    
  function brainLoaded() {
    console.log('pose classification ready!');
    classifyPose();
  }
  
  function classifyPose() {
    if (pose) {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      brain.classify(inputs, gotResult);
    } else {
      setTimeout(classifyPose, 100);
    }
  }
  
  function gotResult(error, results) {
    console.log(results[0].label)
    if (results[0].confidence > 0.75) {
      poseLabel = results[0].label.toUpperCase();
    }
    //console.log(results[0].confidence);
    classifyPose();
  }
  
  
  function gotPoses(poses) {
    if (poses.length > 0) {
      pose = poses[0].pose;
      skeleton = poses[0].skeleton;
    }
  }
  
  function modelLoaded() {
    console.log('poseNet ready');
  }
  
  function draw(p5) {
    p5.push();
    p5.translate(video.width, 0);
    p5.scale(-1, 1);
    p5.image(video, 0, 0, video.width, video.height);
  
    if (pose) {
      for (let i = 0; i < skeleton.length; i++) {
        let a = skeleton[i][0];
        let b = skeleton[i][1];
        p5.strokeWeight(2);
        p5.stroke(0);
  
        p5.line(a.position.x, a.position.y, b.position.x, b.position.y);
      }
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        p5.fill(0);
        p5.stroke(255);
        p5.ellipse(x, y, 16, 16);
      }
    }
    p5.pop();
  
    p5.fill(255, 0, 255);
    p5.noStroke();
    p5.textSize(512);
    p5.textAlign(p5.CENTER, p5.CENTER);
    p5.text(poseLabel, 320, 240);
  }

  return (
    <div className="App" style={{ height: '100vh', display: "flex", justifyContent: 'center', alignItems: 'center' }}>
      <Sketch setup={setup} draw={draw} />
    </div>
  );
}

export default App;