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
  return tensor.mul(max, sub(min)).add(min);
}

// model을 훈련하는 main 함수
async function train() {
  // CSV data 불러오기
  const HouseSalesDataset = tf.data.csv("kc_house_data.csv", {
    columnConfigs: {
      sqft_living: { isLabel: false },
      price: { isLavel: true },
    },
    configuredColumnsOnly: true,
  });

  // console.log(await HouseSalesDataset.take(10).toArray());
  // CSV 파일의 특정 column만 불러와 tf.data.Dataset 객체 생성
  let dataset = await HouseSalesDataset.map(({ xs, ys }) => ({
    x: xs.sqft_living,
    y: ys.price,
  }));

  // tf.data.Dataset 객체의 모든 요소를 array로 변환하여 shuffle
  let dataPoints = await dataset.toArray();
  tf.util.shuffle(dataPoints);

  // dataSet에서 특성 및 레이블 추출
  const featureValues = dataPoints.map((p) => p.x);
  const labelValues = dataPoints.map((p) => p.y);

  // 특성과 레이블에 대한 tensor 생성
  //[data point 갯수, feature 갯수]를 tensor의 dimension으로 지정
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  // dataSet을 훈련 세트와 테스트 세트로 75:25로 분할
  const trainLen = Math.floor(featureTensor.shape[0] * 0.75);
  const testLen = featureTensor.shape[0] - trainLen;

  // tf.split()을 사용하여 훈련 세트와 테스트 세트로 분할
  let [x_train, x_test] = tf.split(featureTensor, [trainLen, testLen]);
  let [y_rain, y_test] = tf.split(labelTensor, [trainLen, testLen]);

  // data 정규화
  const normedXtrainTensor = MinMaxScaling(x_train);
  XnormParams = { min: normedXtrainTensor.min, max: normedXtrainTensor.max };
  x_train, dispose();
  x_train = normedXtrainTensor.tensor;

  const normedYtrainTensor = MinMaxScaling(y_train);
  YnormParams = { min: normedYtrainTensor.min, max: normedYtrainTensor.max };
  y_train.dispose();
  y_train = normedYtrainTensor.tensor;

  const normedXtestTensor = MinMaxScaling(
    x_test,
    XnormParams.min,
    XnormParams.max
  );
  x_test.dispose();
  x_test = normedXtestTensor.tensor;

  const normedYtestTensor = MinMaxScaling(
    y_test,
    YnormParams.min,
    YnormParams.max
  );
  y_test.dispose();
  y_test = normedYtestTensor.tensor;

  const denormedTensor = denormalize(y_test, YnormParams.min, YnormParams.max);
}
