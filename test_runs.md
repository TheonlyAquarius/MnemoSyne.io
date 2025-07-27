(.venv) aqua@luna:~/general/original$ python train_target_model.py
Starting Target CNN training...
Using device: cuda
checkpoints weights will be saved in: checkpoints_weights_cnn
MNIST dataset loaded.
Model, Criterion, Optimizer initialized.
Saved initial random weights to checkpoints_weights_cnn/weights_epoch_0.pth
Epoch [1/10], Batch [100/469], Loss: 0.1434
Epoch [1/10], Batch [200/469], Loss: 0.0498
Epoch [1/10], Batch [300/469], Loss: 0.1387
Epoch [1/10], Batch [400/469], Loss: 0.0791
Epoch [1/10] completed. Average Training Loss: 0.1873
Saved model weights to checkpoints_weights_cnn/weights_epoch_1.pth
Test set: Average loss: 0.0004, Accuracy: 9816/10000 (98.16%)

Epoch [2/10], Batch [100/469], Loss: 0.0292
Epoch [2/10], Batch [200/469], Loss: 0.0362
Epoch [2/10], Batch [300/469], Loss: 0.0924
Epoch [2/10], Batch [400/469], Loss: 0.0571
Epoch [2/10] completed. Average Training Loss: 0.0469
Saved model weights to checkpoints_weights_cnn/weights_epoch_2.pth
Test set: Average loss: 0.0003, Accuracy: 9894/10000 (98.94%)

Epoch [3/10], Batch [100/469], Loss: 0.0422
Epoch [3/10], Batch [200/469], Loss: 0.0265
Epoch [3/10], Batch [300/469], Loss: 0.0273
Epoch [3/10], Batch [400/469], Loss: 0.0306
Epoch [3/10] completed. Average Training Loss: 0.0339
Saved model weights to checkpoints_weights_cnn/weights_epoch_3.pth
Test set: Average loss: 0.0003, Accuracy: 9876/10000 (98.76%)

Epoch [4/10], Batch [100/469], Loss: 0.0044
Epoch [4/10], Batch [200/469], Loss: 0.0189
Epoch [4/10], Batch [300/469], Loss: 0.0074
Epoch [4/10], Batch [400/469], Loss: 0.0194
Epoch [4/10] completed. Average Training Loss: 0.0247
Saved model weights to checkpoints_weights_cnn/weights_epoch_4.pth
Test set: Average loss: 0.0002, Accuracy: 9893/10000 (98.93%)

Epoch [5/10], Batch [100/469], Loss: 0.0157
Epoch [5/10], Batch [200/469], Loss: 0.0018
Epoch [5/10], Batch [300/469], Loss: 0.0030
Epoch [5/10], Batch [400/469], Loss: 0.0158
Epoch [5/10] completed. Average Training Loss: 0.0191
Saved model weights to checkpoints_weights_cnn/weights_epoch_5.pth
Test set: Average loss: 0.0003, Accuracy: 9912/10000 (99.12%)

Epoch [6/10], Batch [100/469], Loss: 0.0099
Epoch [6/10], Batch [200/469], Loss: 0.0181
Epoch [6/10], Batch [300/469], Loss: 0.0103
Epoch [6/10], Batch [400/469], Loss: 0.0060
Epoch [6/10] completed. Average Training Loss: 0.0145
Saved model weights to checkpoints_weights_cnn/weights_epoch_6.pth
Test set: Average loss: 0.0003, Accuracy: 9889/10000 (98.89%)

Epoch [7/10], Batch [100/469], Loss: 0.0020
Epoch [7/10], Batch [200/469], Loss: 0.0310
Epoch [7/10], Batch [300/469], Loss: 0.0045
Epoch [7/10], Batch [400/469], Loss: 0.0082
Epoch [7/10] completed. Average Training Loss: 0.0110
Saved model weights to checkpoints_weights_cnn/weights_epoch_7.pth
Test set: Average loss: 0.0003, Accuracy: 9898/10000 (98.98%)

Epoch [8/10], Batch [100/469], Loss: 0.0006
Epoch [8/10], Batch [200/469], Loss: 0.0001
Epoch [8/10], Batch [300/469], Loss: 0.0134
Epoch [8/10], Batch [400/469], Loss: 0.0311
Epoch [8/10] completed. Average Training Loss: 0.0108
Saved model weights to checkpoints_weights_cnn/weights_epoch_8.pth
Test set: Average loss: 0.0003, Accuracy: 9878/10000 (98.78%)

Epoch [9/10], Batch [100/469], Loss: 0.0014
Epoch [9/10], Batch [200/469], Loss: 0.0120
Epoch [9/10], Batch [300/469], Loss: 0.0125
Epoch [9/10], Batch [400/469], Loss: 0.0355
Epoch [9/10] completed. Average Training Loss: 0.0087
Saved model weights to checkpoints_weights_cnn/weights_epoch_9.pth
Test set: Average loss: 0.0002, Accuracy: 9917/10000 (99.17%)

Epoch [10/10], Batch [100/469], Loss: 0.0052
Epoch [10/10], Batch [200/469], Loss: 0.0025
Epoch [10/10], Batch [300/469], Loss: 0.0010
Epoch [10/10], Batch [400/469], Loss: 0.0030
Epoch [10/10] completed. Average Training Loss: 0.0076
Saved model weights to checkpoints_weights_cnn/weights_epoch_10.pth
Test set: Average loss: 0.0002, Accuracy: 9917/10000 (99.17%)

Target CNN training finished.
Saved final model weights to checkpoints_weights_cnn/weights_epoch_final.pth
(.venv) aqua@luna:~/general/original$ python train_diffusion.py
Initializing reference TargetCNN for dimensions...
Starting Diffusion Model training...
Using device: cuda
Target model flattened dimension: 421642
Diffusion model initialized.
Loading weight checkpoints dataset...
Found 12 weight files.
Found 12 weight files for the checkpoints.
Dataset loaded with 11 pairs.
Criterion and Optimizer initialized.
Starting training loop...
Epoch [1/50], Average Loss: 0.047927
Epoch [2/50], Average Loss: 0.012037
Epoch [3/50], Average Loss: 0.009911
Epoch [4/50], Average Loss: 0.003614
Epoch [5/50], Average Loss: 0.001927
Epoch [6/50], Average Loss: 0.001680
Epoch [7/50], Average Loss: 0.001536
Epoch [8/50], Average Loss: 0.001503
Epoch [9/50], Average Loss: 0.001434
Epoch [10/50], Average Loss: 0.001394
Epoch [11/50], Average Loss: 0.001355
Epoch [12/50], Average Loss: 0.001286
Epoch [13/50], Average Loss: 0.001254
Epoch [14/50], Average Loss: 0.001191
Epoch [15/50], Average Loss: 0.001113
Epoch [16/50], Average Loss: 0.001109
Epoch [17/50], Average Loss: 0.001013
Epoch [18/50], Average Loss: 0.000952
Epoch [19/50], Average Loss: 0.000916
Epoch [20/50], Average Loss: 0.000827
Epoch [21/50], Average Loss: 0.000750
Epoch [22/50], Average Loss: 0.000679
Epoch [23/50], Average Loss: 0.000595
Epoch [24/50], Average Loss: 0.000530
Epoch [25/50], Average Loss: 0.000460
Epoch [26/50], Average Loss: 0.000409
Epoch [27/50], Average Loss: 0.000371
Epoch [28/50], Average Loss: 0.000316
Epoch [29/50], Average Loss: 0.000276
Epoch [30/50], Average Loss: 0.000256
Epoch [31/50], Average Loss: 0.000229
Epoch [32/50], Average Loss: 0.000204
Epoch [33/50], Average Loss: 0.000187
Epoch [34/50], Average Loss: 0.000184
Epoch [35/50], Average Loss: 0.000168
Epoch [36/50], Average Loss: 0.000163
Epoch [37/50], Average Loss: 0.000159
Epoch [38/50], Average Loss: 0.000152
Epoch [39/50], Average Loss: 0.000148
Epoch [40/50], Average Loss: 0.000140
Epoch [41/50], Average Loss: 0.000142
Epoch [42/50], Average Loss: 0.000147
Epoch [43/50], Average Loss: 0.000139
Epoch [44/50], Average Loss: 0.000138
Epoch [45/50], Average Loss: 0.000142
Epoch [46/50], Average Loss: 0.000133
Epoch [47/50], Average Loss: 0.000132
Epoch [48/50], Average Loss: 0.000125
Epoch [49/50], Average Loss: 0.000132
Epoch [50/50], Average Loss: 0.000123
Trained diffusion model saved to diffusion_optimizer.pth
(.venv) aqua@luna:~/general/original$ ls
README.md    data                diffusion_optimizer.pth  evaluate_generalization.py     target_cnn.py       train_target_model.py
__pycache__  diffusion_model.py  evaluate_diffusion.py    evaluate_generated_weights.py  train_diffusion.py  checkpoints_weights_cnn
(.venv) aqua@luna:~/general/original$ python evaluate_diffusion.py
Starting evaluation of diffusion-generated checkpoints...
Using device: cuda
Target CNN: flat dimension = 421642
Diffusion model loaded from diffusion_optimizer.pth
Will generate a checkpoints of 11 steps.
Generating checkpoints with diffusion model for 11 steps...
  Generated step 10/11
  Generated step 11/11

Evaluating performance along the generated checkpoints...
Step 0 (Initial Random Weights): Accuracy = 11.62%, Avg Loss = 591.7755
Generated Step 1/11: Accuracy = 99.13%, Avg Loss = 145.6099
Generated Step 2/11: Accuracy = 99.18%, Avg Loss = 55.5481
Generated Step 3/11: Accuracy = 99.21%, Avg Loss = 25.9044
Generated Step 4/11: Accuracy = 99.21%, Avg Loss = 15.4776
Generated Step 5/11: Accuracy = 99.21%, Avg Loss = 11.1127
Generated Step 6/11: Accuracy = 99.22%, Avg Loss = 8.9604
Generated Step 7/11: Accuracy = 99.24%, Avg Loss = 7.7551
Generated Step 8/11: Accuracy = 99.26%, Avg Loss = 7.0136
Generated Step 9/11: Accuracy = 99.26%, Avg Loss = 6.5302
Generated Step 10/11: Accuracy = 99.26%, Avg Loss = 6.2055
Generated Step 11/11: Accuracy = 99.24%, Avg Loss = 5.9860

Evaluating performance along the original CNN training checkpoints...
Original CNN Epoch 0: Accuracy = 11.62%, Avg Loss = 591.7755
Original CNN Epoch 1: Accuracy = 98.16%, Avg Loss = 14.4651
Original CNN Epoch 2: Accuracy = 98.94%, Avg Loss = 8.2082
Original CNN Epoch 3: Accuracy = 98.76%, Avg Loss = 9.3712
Original CNN Epoch 4: Accuracy = 98.93%, Avg Loss = 7.9326
Original CNN Epoch 5: Accuracy = 99.12%, Avg Loss = 8.2732
Original CNN Epoch 6: Accuracy = 98.89%, Avg Loss = 8.2651
Original CNN Epoch 7: Accuracy = 98.98%, Avg Loss = 8.8551
Original CNN Epoch 8: Accuracy = 98.78%, Avg Loss = 9.1651
Original CNN Epoch 9: Accuracy = 99.17%, Avg Loss = 7.3522
Original CNN Epoch 10: Accuracy = 99.17%, Avg Loss = 7.2921
Warning: Original weight file checkpoints_weights_cnn/weights_epoch_11.pth not found. Skipping.

Plotting results...
Traceback (most recent call last):
  File "/home/aqua/general/original/evaluate_diffusion.py", line 254, in <module>
    evaluate_diffusion_generated_checkpoints(
  File "/home/aqua/general/original/evaluate_diffusion.py", line 205, in evaluate_diffusion_generated_checkpoints
    plt.figure(figsize=(12, x3))
                            ^^
NameError: name 'x3' is not defined
(.venv) aqua@luna:~/general/original$ nano evaluate_diffusion.py
(.venv) aqua@luna:~/general/original$ python evaluate_diffusion.py
Starting evaluation of diffusion-generated checkpoints...
Using device: cuda
Target CNN: flat dimension = 421642
Diffusion model loaded from diffusion_optimizer.pth
Will generate a checkpoints of 11 steps.
Generating checkpoints with diffusion model for 11 steps...
  Generated step 10/11
  Generated step 11/11

Evaluating performance along the generated checkpoints...
Step 0 (Initial Random Weights): Accuracy = 11.62%, Avg Loss = 591.7755
Generated Step 1/11: Accuracy = 99.13%, Avg Loss = 145.6099
Generated Step 2/11: Accuracy = 99.18%, Avg Loss = 55.5481
Generated Step 3/11: Accuracy = 99.21%, Avg Loss = 25.9044
Generated Step 4/11: Accuracy = 99.21%, Avg Loss = 15.4776
Generated Step 5/11: Accuracy = 99.21%, Avg Loss = 11.1127
Generated Step 6/11: Accuracy = 99.22%, Avg Loss = 8.9604
Generated Step 7/11: Accuracy = 99.24%, Avg Loss = 7.7551
Generated Step 8/11: Accuracy = 99.26%, Avg Loss = 7.0136
Generated Step 9/11: Accuracy = 99.26%, Avg Loss = 6.5302
Generated Step 10/11: Accuracy = 99.26%, Avg Loss = 6.2055
Generated Step 11/11: Accuracy = 99.24%, Avg Loss = 5.9860

Evaluating performance along the original CNN training checkpoints...
Original CNN Epoch 0: Accuracy = 11.62%, Avg Loss = 591.7755
Original CNN Epoch 1: Accuracy = 98.16%, Avg Loss = 14.4651
Original CNN Epoch 2: Accuracy = 98.94%, Avg Loss = 8.2082
Original CNN Epoch 3: Accuracy = 98.76%, Avg Loss = 9.3712
Original CNN Epoch 4: Accuracy = 98.93%, Avg Loss = 7.9326
Original CNN Epoch 5: Accuracy = 99.12%, Avg Loss = 8.2732
Original CNN Epoch 6: Accuracy = 98.89%, Avg Loss = 8.2651
Original CNN Epoch 7: Accuracy = 98.98%, Avg Loss = 8.8551
Original CNN Epoch 8: Accuracy = 98.78%, Avg Loss = 9.1651
Original CNN Epoch 9: Accuracy = 99.17%, Avg Loss = 7.3522
Original CNN Epoch 10: Accuracy = 99.17%, Avg Loss = 7.2921
Warning: Original weight file checkpoints_weights_cnn/weights_epoch_11.pth not found. Skipping.

Plotting results...
Plot saved to diffusion_evaluation_plot.png
Evaluation finished.
(.venv) aqua@luna:~/general/original$ python evaluate_generalization.py
Starting evaluation of diffusion-generated checkpoints...
Using device: cuda
Target CNN: flat dimension = 421642
Diffusion model loaded from diffusion_optimizer.pth

--- Generating new random weights to test generalization ---
New random weights generated and flattened.
Will generate a checkpoints of 11 steps.
Generating checkpoints with diffusion model for 11 steps...
  Generated step 10/11
  Generated step 11/11

Evaluating performance along the generated checkpoints...
Step 0 (Initial Random Weights): Accuracy = 14.97%, Avg Loss = 589.0321
Generated Step 1/11: Accuracy = 96.77%, Avg Loss = 538.1667
Generated Step 2/11: Accuracy = 99.01%, Avg Loss = 315.7714
Generated Step 3/11: Accuracy = 99.12%, Avg Loss = 115.9179
Generated Step 4/11: Accuracy = 99.20%, Avg Loss = 40.2107
Generated Step 5/11: Accuracy = 99.21%, Avg Loss = 18.6301
Generated Step 6/11: Accuracy = 99.21%, Avg Loss = 11.6120
Generated Step 7/11: Accuracy = 99.22%, Avg Loss = 8.8007
Generated Step 8/11: Accuracy = 99.24%, Avg Loss = 7.4549
Generated Step 9/11: Accuracy = 99.26%, Avg Loss = 6.7229
Generated Step 10/11: Accuracy = 99.27%, Avg Loss = 6.2903
Generated Step 11/11: Accuracy = 99.25%, Avg Loss = 6.0225

Plotting results...
Plot saved to diffusion_evaluation_plot.png
Save the generated weights from this checkpoints? (yes/no): yes
Saved 12 weight files to 'generalized_checkpoints_weights'.
Evaluation finished.
(.venv) aqua@luna:~/general/original$ ls
README.md    data                           diffusion_model.py       evaluate_diffusion.py       evaluate_generated_weights.py   target_cnn.py       train_target_model.py
__pycache__  diffusion_evaluation_plot.png  diffusion_optimizer.pth  evaluate_generalization.py  generalized_checkpoints_weights  train_diffusion.py  checkpoints_weights_cnn
(.venv) aqua@luna:~/general/original$ python evaluate_generalization.py
Starting evaluation of diffusion-generated checkpoints...
Using device: cuda
Target CNN: flat dimension = 421642
Diffusion model loaded from diffusion_optimizer.pth

--- Generating new random weights to test generalization ---
New random weights generated and flattened.
Will generate a checkpoints of 11 steps.
Generating checkpoints with diffusion model for 11 steps...
  Generated step 10/11
  Generated step 11/11

Evaluating performance along the generated checkpoints...
Step 0 (Initial Random Weights): Accuracy = 9.09%, Avg Loss = 590.1954
Generated Step 1/11: Accuracy = 97.57%, Avg Loss = 525.8835
Generated Step 2/11: Accuracy = 99.03%, Avg Loss = 293.0688
Generated Step 3/11: Accuracy = 99.13%, Avg Loss = 105.2084
Generated Step 4/11: Accuracy = 99.20%, Avg Loss = 37.3612
Generated Step 5/11: Accuracy = 99.20%, Avg Loss = 17.8595
Generated Step 6/11: Accuracy = 99.21%, Avg Loss = 11.3672
Generated Step 7/11: Accuracy = 99.22%, Avg Loss = 8.7112
Generated Step 8/11: Accuracy = 99.24%, Avg Loss = 7.4189
Generated Step 9/11: Accuracy = 99.26%, Avg Loss = 6.7077
Generated Step 10/11: Accuracy = 99.27%, Avg Loss = 6.2837
Generated Step 11/11: Accuracy = 99.25%, Avg Loss = 6.0197

Plotting results...
Plot saved to diffusion_evaluation_plot.png
Save the generated weights from this checkpoints? (yes/no): yes
Saved 12 weight files to 'generalized_checkpoints_weights'.
Evaluation finished.
