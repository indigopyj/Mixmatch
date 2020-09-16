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

## Comparison with paper
### Cifar10

|#Labels|250|500|1000|2000|4000|
|-------|---|---|----|----|----|
|Paper| 88.92 ± 0.87|90.35 ± 0.94|92.25 ± 0.32|92.97 ± 0.15|93.76 ± 0.06|
|This code|88.990|90.473|90.593|92.625|93.443|


