# kernel-k-nearest-neighbor-classifier

implement kernel k-nearest neighbor classifier. Recall that in 1NN
classifier (𝑘 = 1), we just need to compute the Euclidean distance of a test instance to all the training
instances, find the closest training instance, and find its label. The label of the nest instance will be
identical to its nearest neighbor. This can be kernelized by observing that:

‖𝑥𝑖 − 𝑥𝑖′ ‖2 2 = < 𝑥𝑖 , 𝑥𝑖 > + < 𝑥𝑖′ , 𝑥𝑖′ > − 2 < 𝑥𝑖, 𝑥𝑖′ > = 𝐾(𝑥𝑖, 𝑥𝑖 ) + 𝐾(𝑥𝑖′ , 𝑥𝑖′ ) − 2𝐾(𝑥𝑖, 𝑥𝑖′ )

This allows us to apply the nearest neighbor classifier to structured data objects.

Implement the KNN classifier and kernel KNN classifier with Linear, RBF (tune the 𝜎 parameter with
cross-validation), Polynomial (𝑑 = 1), Polynomial (𝑑 = 2), and Polynomial (𝑑 = 3) kernels. Report the
accuracy of classification for each data set with each classifier and compare the results. Split the data
set into 70% and 30% for training and testing parts. You should report the mean of accuracies for 10
individual runs. Report the running time of your code (in seconds).
