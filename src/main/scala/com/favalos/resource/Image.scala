package com.favalos.resource

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File

import javax.imageio.ImageIO

object Image {

  def write(name: String, arr: Array[Array[Float]]) = {

    val out = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)

    for(w <- 0 until arr.size) {
      for (h <- 0 until arr(w).size) {

        val c = Math.max(Math.min(arr(w)(h) * 255, 255), 0).toInt
        out.setRGB(h, w, new Color(c, c, c).getRGB)
      }
    }

    ImageIO.write(out, "jpg", new File(s"${name}.jpg"))
  }

}
