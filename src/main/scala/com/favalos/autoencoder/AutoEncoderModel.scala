package com.favalos.autoencoder

import com.favalos.resource.Image
import org.apache.mxnet.module.Module
import org.apache.mxnet.optimizer.RMSProp
import org.apache.mxnet.{Context, DataIter, MSE, NDArray, Symbol, Xavier}
import org.slf4j.LoggerFactory

/**
  * An Autoencoder implementation using FullyConnected Layers and RELU activations.
  *
  * @author Fernando Avalos
  */

class AutoEncoderModel(val dims: Array[Int]) {

  private val logger = LoggerFactory.getLogger(classOf[AutoEncoderModel])

  def encoder(): Symbol = {

    val encoderNetwork = createNetwork("enc", dims)

    encoderNetwork
  }

  def decoder(encoded: Symbol): Symbol = {

    val reverseDims = dims.reverse

    val decoderNetwork = createNetwork("dec", reverseDims, encoded)

    decoderNetwork
  }

  private[autoencoder] def createNetwork(prefix: String, dims: Array[Int], data: Symbol = Symbol.Variable("data")): Symbol = {

    val network = dims.zip(1 to dims.length).foldLeft(data){

      ( layer, idx ) =>
        val fc = Symbol.FullyConnected(s"${prefix}_dl${idx._2}")()(Map("data" -> layer, "num_hidden" -> idx._1))
        val act = Symbol.Activation(s"${prefix}_act${idx._2}")()(Map("data" -> fc, "act_type" -> "relu"))
        act
    }

    network
  }

  def train(dataIter: DataIter, epoch: Int, ctx: Context = Context.cpu(), lr: Float = 0.00001f) = {

    val except = Set("data", "label")
    val label = Symbol.Variable("label")

    val encoderSymbol = encoder()
    val decoderSymbol = decoder(encoderSymbol)

    def loss(output: Symbol, label: Symbol): Symbol = {
      Symbol.abs("loss")()(Map("data" -> (output - label)))
    }
    val decoded = Symbol.MakeLoss("decoded")()(Map("data" -> loss(decoderSymbol, label)))

    val initializer = new Xavier()

    val (decInputShape, _, _) = decoded.inferShape(dataIter.provideData + ("label" -> dataIter.provideData("data")))

    val argsDict = decoded.listArguments().zip(decInputShape.map(s => NDArray.empty(s, ctx))).toMap

    val gradDict = decoderSymbol.listArguments()
      .zip(decInputShape).filterNot(v => except.contains(v._1)).map {
      case (key, shape) => key -> NDArray.empty(shape, ctx)
    }.toMap

    argsDict.foreach {
      case (key, ndArray) =>
        if (!except.contains(key))
          initializer.initWeight(key, ndArray)
    }

    val executor = decoded.bind(ctx, argsDict, gradDict)
    val opt = new RMSProp(learningRate = lr, wd = 0.0001f)

    val paramGrads = gradDict.toList.zipWithIndex.map {
      case ((key, grad), idx) => (idx, key, grad, opt.createState(idx, argsDict(key)))
    }
    val metric = new MSE()

    for (i <- 0 until epoch) {
      metric.reset()
      dataIter.reset()

      while (dataIter.hasNext) {

        val dataBatch = dataIter.next()
        argsDict("data").set(dataBatch.data(0))
        argsDict("label").set(dataBatch.data(0))

        executor.forward(isTrain = true)
        executor.backward()

        paramGrads.foreach {
          case (idx, key, grad, state) =>
            opt.update(idx, argsDict(key), grad, state)
        }

        metric.update(dataBatch.data, executor.outputs)
      }

      logger.debug(s"Iteration: ${i} MSE metric is: ${metric.get._2(0)}")

    }

    argsDict
  }

  def testModel(dataIter: DataIter, argsDict: Map[String, NDArray] ) = {

    val dataShape = dataIter.provideData("data")

    def encoderSymbol = encoder()
    def decoderSymbol = decoder(encoderSymbol)

    val decoderModule = new Module( decoderSymbol)
    decoderModule.bind( dataIter.provideData , None, false)
    decoderModule.initParams(argParams = argsDict)

    val dataBatch = dataIter.next
    val encodedResponse = decoderModule.predict(dataBatch)

    val sourceArr = dataBatch.data(0).toArray.grouped(dataShape(2) * dataShape(3)).toArray
    val genArr = encodedResponse(0).toArray.grouped(dataShape(2) * dataShape(3)).toArray

    for(i <- 0 until sourceArr.length) {

      Image.write(s"source_idx_${i}", sourceArr(i).grouped(dataShape(2)).toArray)
      Image.write(s"decoded_idx_${i}", genArr(i).grouped(dataShape(2)).toArray)
    }

  }
}


