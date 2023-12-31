// model 훈련의 에폭 수 지정
const EPOCHS = 10;
let XnormParams, YnormParams, model;

// TensorFlow.js Visor를 토글하는 함수
function toggleVisor() {
  tfvis.visor().toggle();
}

// data를 Min-Max 스케일링을 이용하여 정규화하는 함수
function MinMaxScaling(tensor, prevMin = null, prevMax = null) {
  const min = prevMin || tensor.min();
  const max = prevMax || tensor.max();
  const normedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor: normedTensor,
    min,
    max,
  };
}

// data를 denormalize 하는 함수
function denormalize(tensor, min, max) {
  return tensor.mul(max.sub(min)).add(min);
}

// model을 훈련하는 main 함수
async function train() {
  // CSV 데이터 불러오기
  const HouseSalesDataset = tf.data.csv("kc_house_data.csv", {
    columnConfigs: {
      sqft_living: { isLabel: false },
      price: { isLabel: true },
    },
    configuredColumnsOnly: true,
  });

  // console.log(await HouseSalesDataset.take(10).toArray());
  // csv 파일의 특정 column만 불러와 tf.data.Dataset 객체 생성
  let dataset = await HouseSalesDataset.map(({ xs, ys }) => ({
    x: xs.sqft_living,
    y: ys.price,
  }));

  // tf.data.Dataset 객체의 모든 요소를 array로 변환하여 shuffle
  let dataPoints = await dataset.toArray();
  tf.util.shuffle(dataPoints);

  // 데이터를 tfvis를 사용하여 그래프로 시각화
  let surface = { name: "면적 vs 가격", tab: "Data" };
  let data = { values: [dataPoints] };
  let opts = { xLabel: "면적", yLabel: "가격" };
  tfvis.render.scatterplot(surface, data, opts);

  // 데이터셋에서 특성 및 레이블 추출
  const featureValues = dataPoints.map((p) => p.x);
  const labelValues = dataPoints.map((p) => p.y);

  // 특성과 레이블에 대한 텐서 생성
  //[data point 갯수, feature 갯수]를 tensor의 dimension으로 지정
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  // 데이터셋을 훈련 세트와 테스트 세트로 75:25로 분할
  const trainLen = Math.floor(featureTensor.shape[0] * 0.75);
  const testLen = featureTensor.shape[0] - trainLen;

  //tf.split()을 사용하여 훈련 세트와 테스트 세트로 분할
  let [X_train, X_test] = tf.split(featureTensor, [trainLen, testLen]);
  let [y_train, y_test] = tf.split(labelTensor, [trainLen, testLen]);

  // 데이터 정규화
  const normedXtrainTensor = MinMaxScaling(X_train);
  XnormParams = { min: normedXtrainTensor.min, max: normedXtrainTensor.max };
  X_train.dispose();
  X_train = normedXtrainTensor.tensor;

  const normedYtrainTensor = MinMaxScaling(y_train);
  YnormParams = { min: normedYtrainTensor.min, max: normedYtrainTensor.max };
  y_train.dispose();
  y_train = normedYtrainTensor.tensor;

  const normedXtestTensor = MinMaxScaling(
    X_test,
    XnormParams.min,
    XnormParams.max
  );
  X_test.dispose();
  X_test = normedXtestTensor.tensor;

  const normedyTestTensor = MinMaxScaling(
    y_test,
    YnormParams.min,
    YnormParams.max
  );
  y_test.dispose();
  y_test = normedyTestTensor.tensor;

  const denormedTensor = denormalize(y_test, YnormParams.min, YnormParams.max);

  // 정규화된 데이터 시각화
  const normedPoints = [];
  const normedFeatureArr = X_train.arraySync();
  const normedLabelArr = y_train.arraySync();

  for (let i = 0; i < normedFeatureArr.length; i++) {
    normedPoints.push({ x: normedFeatureArr[i], y: normedLabelArr[i] });
  }

  surface = { name: "Normalized - 면적 vs 가격", tab: "Data" };
  data = { values: [normedPoints] };
  opts = { xLabel: "면적", yLabel: "가격" };
  tfvis.render.scatterplot(surface, data, opts);

  // 모델 구성
  model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 10, inputShape: [1], activation: "relu" })
  );
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));
  model.compile({ loss: "meanSquaredError", optimizer: tf.train.adam(0.01) });
  model.summary();
  tfvis.show.modelSummary({ name: "model summary", tab: "model" }, model);

  // 훈련 중 상태 시각화를 위한 콜백 함수
  const container = { name: "Training Performance" };
  const metrics = ["loss", "val_loss"];
  const { onEpochEnd, onBatchEnd } = tfvis.show.fitCallbacks(
    container,
    metrics
  );

  // 모델 훈련
  const history = await model.fit(X_train, y_train, {
    epochs: EPOCHS,
    batchSize: 32,
    validationData: [X_test, y_test],
    callbacks: { onEpochEnd, onBatchEnd },
  });

  // train, validation loss 출력
  console.log(history.history.loss);
  const trainLoss = history.history.loss.pop();
  const validationLoss = history.history.val_loss.pop();
  console.log(
    `Train Loss = ${trainLoss} \nValidation Loss = ${validationLoss}`
  );

  //train 완료 후 prediction button 활성화
  document.getElementById("predict-button").removeAttribute("disabled");
}

// // 예측 함수
async function predict() {
  const predictionInput = parseInt(
    document.getElementById("predict-input").value
  );
  // 예측 입력값이 숫자가 아니라면 알림을 띄움
  if (isNaN(predictionInput)) {
    alert("숫자를 입력하세요");
  } else {
    tf.tidy(() => {
      const inputTensor = tf.tensor1d([predictionInput]);
      const normedInput = MinMaxScaling(
        inputTensor,
        XnormParams.min,
        XnormParams.max
      );
      const prediction = model.predict(normedInput.tensor);

      const denormedPrediction = denormalize(
        prediction,
        YnormParams.min,
        YnormParams.max
      );
      const output = denormedPrediction.dataSync()[0];
      document.getElementById(
        "predict-output"
      ).innerHTML = `Predicted Price <br> ${output
        .toFixed(2)
        .toLocaleString()}`;

      // 예측 결과를 시각화하기 위한 라인 그리기
      const [xs, ys] = tf.tidy(() => {
        const normedXs = tf.linspace(0, 1, 100);
        const normedYs = model.predict(normedXs.reshape([100, 1]));
        const denormedXs = denormalize(
          normedXs,
          XnormParams.min,
          XnormParams.max
        );
        const denormedYs = denormalize(
          normedYs,
          YnormParams.min,
          YnormParams.max
        );
        return [denormedXs.dataSync(), denormedYs.dataSync()];
      });

      const pointsLine = Array.from(xs).map((x, index) => ({
        x: x,
        y: ys[index],
      }));

      surface = { name: "Predict Line", tab: "Data" };
      data = { values: pointsLine, series: ["Predictions "] };
      opts = { xLabel: "Square Feet", yLabel: "Price" };
      tfvis.render.scatterplot(surface, data, opts);
    });
  }
}
