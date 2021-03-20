let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "Y";

export default function sketch(p) {
  
  p.setup = function () {
    p.createCanvas(640, 480);
    video = p.createCapture(p.VIDEO);
    video.hide();
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on('pose', gotPoses);

    let options = {
      inputs: 34,
      outputs: 4,
      task: 'classification',
      debug: true
    }
    brain = ml5.neuralNetwork(options);
    const modelInfo = {
      model: './model3/model.json',
      metadata: './model3/model_meta.json',
      weights: './model3/model.weights.bin',
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

  p.draw = function() {
    p.push();
    p.translate(video.width, 0);
    p.scale(-1, 1);
    p.image(video, 0, 0, video.width, video.height);

    if (pose) {
      for (let i = 0; i < skeleton.length; i++) {
        let a = skeleton[i][0];
        let b = skeleton[i][1];
        p.strokeWeight(2);
        p.stroke(0);

        p.line(a.position.x, a.position.y, b.position.x, b.position.y);
      }
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        p.fill(0);
        p.stroke(255);
        p.ellipse(x, y, 16, 16);
      }
    }
    p.pop();

    p.fill(255, 0, 255);
    p.noStroke();
    p.textSize(512);
    p.textAlign(p.CENTER, p.CENTER);
    p.text(poseLabel, video.width / 2, video.height / 2);
  }
}
