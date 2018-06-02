package com.favalos.autoencoder

import java.io.File

import org.apache.mxnet.{DataIter, IO}

/**
  * A simple example of training an autoencoder on MNIST.
  *
  * This sample code will execute the following actions:
  * 1. Train the AutoEncoder using the MNIST train dataset.
  * 2. Test the AutoEncoder getting a batch from MNIST test and generate images.
  *
  * @author Fernando Avalos
  */
object MNISTAutoencoder {

  def main(args: Array[String]): Unit = {

    val localDir = new File(".").getCanonicalPath

    val encoderLayersSize = Array(784, 512, 256, 128)
    val dataIter = loadDataMNIST(localDir = localDir)
    val testDataIter = loadTestMNIST(localDir = localDir)

    val autoEncoder = new AutoEncoderModel(encoderLayersSize)
    val argsDict = autoEncoder.train(dataIter, 5)

    autoEncoder.testModel(testDataIter, argsDict)


  }

  def loadDataMNIST(batchSize: Int = 100, localDir: String): DataIter = {

    val trainDataIter = IO.MNISTIter(Map(
      "image" -> s"${localDir}/src/main/resources/data/train-images-idx3-ubyte",
      "label" -> s"${localDir}/src/main/resources/data/train-labels-idx1-ubyte",
      "label_name" -> "label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "1", "silent" -> "0", "seed" -> "10"))

    trainDataIter
  }

  def loadTestMNIST(batchSize: Int = 10, localDir: String): DataIter = {

    val testDataIter = IO.MNISTIter(Map(
      "image" -> s"${localDir}/src/main/resources/data/t10k-images-idx3-ubyte",
      "label" -> s"${localDir}/src/main/resources/data/t10k-labels-idx1-ubyte",
      "label_name" -> "label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0", "seed" -> "10"))

    testDataIter
  }

}
