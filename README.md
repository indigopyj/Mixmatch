# MixMatch

![image](https://user-images.githubusercontent.com/17904547/92840923-a869f780-f41c-11ea-848a-22816dede0ae.png)

- Referenced Codes: 
   - https://github.com/google-research/mixmatch
   - https://github.com/YU1ut/MixMatch-pytorch
   


## Hyperparameters Setting

- learning rate = 0.002
- Decay rate of Exponential moving average = 0.999
- Weight decay = 0.00004
- number of epochs = 1000
- batch size = 64
- T = 0.5
- K = 2
- lambda_u = 75
- alpha = 0.75

※ Caution : I trained only 1 time.(Original paper trained total 5 times with different random seed.)

## Results

### Cifar10 250 examples

<img width="500" alt="스크린샷 2020-09-14 오후 2 31 06" src="https://user-images.githubusercontent.com/17904547/93049220-8af69100-f69b-11ea-9d11-06b3833c08af.png">

- **Test accuracy : 88.990**

### Cifar10 500 examples

<img width="500" alt="스크린샷 2020-09-14 오후 2 31 17" src="https://user-images.githubusercontent.com/17904547/93049225-8e8a1800-f69b-11ea-9ae7-0e856369d65b.png">

- **Test accuracy : 90.473**

### Cifar10 1000 examples

<img width="500" alt="스크린샷 2020-09-14 오후 2 31 27" src="https://user-images.githubusercontent.com/17904547/93049227-8fbb4500-f69b-11ea-8303-4c4dc5c699bb.png">

- **Test accuracy : 90.593**

### Cifar10 2000 examples
<img width="500" src="https://user-images.githubusercontent.com/17904547/93293966-434f4100-f824-11ea-8a5a-c99d1c110838.png">

- **Test accuracy : 92.625**

### Cifar10 4000 examples

<img width="500" src="https://user-images.githubusercontent.com/17904547/93294014-61b53c80-f824-11ea-8a65-7d112e55972d.png">

- **Test accuracy : 93.443**


## Comparison with paper

|#Labels|250|500|1000|2000|4000|
|-------|---|---|----|----|----|
|Paper| 88.92 ± 0.87|90.35 ± 0.94|92.25 ± 0.32|92.97 ± 0.15|93.76 ± 0.06|
|This code|88.990|90.473|90.593|92.625|93.443|


