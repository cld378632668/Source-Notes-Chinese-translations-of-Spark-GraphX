/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.graphx.lib

import scala.reflect.ClassTag
import scala.language.postfixOps

import org.apache.spark.Logging
import org.apache.spark.graphx._

/**
  * 计算一张图中所有顶点的重要程度,进而对他们进行排名.
  *  这是GraphX提供的用Pregel的模型改进后产生的图算法，通常我们在进行使用PageRank的代码编写时并不涉及去改动这份源码，
  *  而是直接调用：graph.pageRank(0.0001)
  */

/**
 * PageRank algorithm implementation. There are two implementations of PageRank implemented.
 *PageRank的两种实现方式.
 * The first implementation uses the standalone [[Graph]] interface and runs PageRank
 * for a fixed number of iterations:
 * 第一种（静态实现）: 使用standalone [[Graph]]接口,在调用时提供一个参数number，用于指定迭代次数，
  * 即无论结果如何，该算法在迭代number次后停止计算，返回图结果。
 * {{{
 * var PR = Array.fill(n)( 1.0 )
 * val oldPR = Array.fill(n)( 1.0 )
 * for( iter <- 0 until numIter ) {
 *   swap(oldPR, PR)
 *   for( i <- 0 until n ) {
 *     PR[i] = alpha + (1 - alpha) * inNbrs[i].map(j => oldPR[j] / outDeg[j]).sum
 *   }
 * }
 * }}}
 *
 * The second implementation uses the [[Pregel]] interface and runs PageRank until
 * convergence:
 * 第二种：（动态）在调用时提供一个参数tol，用于指定前后两次迭代的结果差值应小于tol，
  * 以达到最终收敛的效果时才停止计算，返回图结果。
 * {{{
 * var PR = Array.fill(n)( 1.0 )
 * val oldPR = Array.fill(n)( 0.0 )
 * while( max(abs(PR - oldPr)) > tol ) {
 *   swap(oldPR, PR)
 *   for( i <- 0 until n if abs(PR[i] - oldPR[i]) > tol ) {
 *     PR[i] = alpha + (1 - \alpha) * inNbrs[i].map(j => oldPR[j] / outDeg[j]).sum
 *   }
 * }
 * }}}
 *
 * `alpha` is the random reset probability (typically 0.15), `inNbrs[i]` is the set of
 * neighbors which link to `i` and `outDeg[j]` is the out degree of vertex `j`.
 *
 * Note that this is not the "normalized" PageRank and as a consequence pages that have no
 * inlinks will have a PageRank of alpha.
 */
object PageRank extends Logging {


  /**
   * Run PageRank for a fixed number of iterations returning a graph
   * with vertex attributes containing the PageRank and edge
   * attributes the normalized edge weight.
    * 以固定的迭代次数运行PageRank算法,以图的形式返回,包括最终的顶点值(pagerank值)
    * 和标准化的边值(权重值),进而得到最终的排名结果.
   * @tparam VD the original vertex attribute (not used)
    *            顶点的属性类型(不需要用户指定,由你传入的图的属性决定)
   * @tparam ED the original edge attribute (not used)
   *              边的属性类型(不需要用户指定,由你传入的图的属性决定)
   * @param graph the graph on which to compute PageRank
    *              进行PageRank计算的图模型
   * @param numIter the number of iterations of PageRank to run
    *                迭代次数
   * @param resetProb the random reset probability (alpha)
    *                  随机重置的概率,通常都是0.15
   * @return the graph containing with each vertex containing the PageRank and each edge
   *         containing the normalized weight.
    *         以图的形式返回,包括最终的顶点值(pagerank值)和标准化的边值(权重值),
    *         进而得到最终的排名结果
   */
  def run[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED], numIter: Int,
    resetProb: Double = 0.15): Graph[Double, Double] =
  {
    runWithOptions(graph, numIter, resetProb)
  }

  /**
   * Run PageRank for a fixed number of iterations returning a graph
   * with vertex attributes containing the PageRank and edge
   * attributes the normalized edge weight.
   *
    * withOptions : 可以个性化定义srcId
    *
   * @tparam VD the original vertex attribute (not used)
   * @tparam ED the original edge attribute (not used)
   *
   * @param graph the graph on which to compute PageRank
    *              进行PageRank计算的图模型
   * @param numIter the number of iterations of PageRank to run
    *                迭代次数
   * @param resetProb the random reset probability (alpha)
    *                  随机重置的概率,通常都是0.15
   * @param srcId the source vertex for a Personalized Page Rank (optional)
    *              个性化的顶点Id值
   * @return the graph containing with each vertex containing the PageRank and each edge
   *         containing the normalized weight.
    *         以图的形式返回,包括最终的顶点值(pagerank值)和标准化的边值(权重值)
   */
  def runWithOptions[VD: ClassTag, ED: ClassTag](
      graph: Graph[VD, ED], numIter: Int, resetProb: Double = 0.15,
      srcId: Option[VertexId] = None): Graph[Double, Double] =
  {
    //srcId是否被定义,即是否个性化,返回Boolean值
    val personalized = srcId isDefined
    //获取srcId的具体值,如果为空,则赋值为-1L
    val src: VertexId = srcId.getOrElse(-1L)

    // Initialize the PageRank graph with each edge attribute having
    // weight 1/outDegree and each vertex with attribute resetProb.
    // 下面这段代码用于初始化PageRank图模型rankGraph，具体内容是
    // 赋予每条边属性为值“1/该边的出度数”,赋予每个顶点属性为resetProb的值。
    // When running personalized pagerank, only the source vertex
    // has an attribute resetProb. All others are set to 0.
    // 当运行personlized PageRank时,仅仅是出发顶点使用resetProb作为属性,
    // 其他所有顶点的属性被设置为0.
    var rankGraph: Graph[Double, Double] = graph
      // Associate the degree with each vertex
      // 将出度数与每个顶点的属性值关联(给每个顶点添加出度这个属性)
      .outerJoinVertices(graph.outDegrees) { (vid, vdata, deg) => deg.getOrElse(0) }
      // Set the weight on the edges based on the degree
      //基于度数为边设置权重
      .mapTriplets( e => 1.0 / e.srcAttr, TripletFields.Src )
      // Set the vertex attributes to the initial pagerank values
      //设置每个顶点的初始属性为initial pagerank values
      .mapVertices { (id, attr) =>
        if (!(id != src && personalized)) resetProb else 0.0
      }

    def delta(u: VertexId, v: VertexId): Double = { if (u == v) 1.0 else 0.0 }

    var iteration = 0
    var prevRankGraph: Graph[Double, Double] = null
    while (iteration < numIter) {
      rankGraph.cache()

      // Compute the outgoing rank contributions of each vertex, perform local preaggregation, and
      // do the final aggregation at the receiving vertices. Requires a shuffle for aggregation.
      val rankUpdates = rankGraph.aggregateMessages[Double](
        ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr), _ + _, TripletFields.Src)

      // Apply the final rank updates to get the new ranks, using join to preserve ranks of vertices
      // that didn't receive a message. Requires a shuffle for broadcasting updated ranks to the
      // edge partitions.
      prevRankGraph = rankGraph
      val rPrb = if (personalized) {
        (src: VertexId , id: VertexId) => resetProb * delta(src, id)
      } else {
        (src: VertexId, id: VertexId) => resetProb
      }

      rankGraph = rankGraph.joinVertices(rankUpdates) {
        (id, oldRank, msgSum) => rPrb(src, id) + (1.0 - resetProb) * msgSum
      }.cache()

      rankGraph.edges.foreachPartition(x => {}) // also materializes rankGraph.vertices
      logInfo(s"PageRank finished iteration $iteration.")
      prevRankGraph.vertices.unpersist(false)
      prevRankGraph.edges.unpersist(false)

      iteration += 1
    }

    rankGraph
  }

  /**
   * Run a dynamic version of PageRank returning a graph with vertex attributes containing the
   * PageRank and edge attributes containing the normalized edge weight.
   *
   * @tparam VD the original vertex attribute (not used)
   * @tparam ED the original edge attribute (not used)
   *
   * @param graph the graph on which to compute PageRank
   * @param tol the tolerance allowed at convergence (smaller => more accurate).
   * @param resetProb the random reset probability (alpha)
   *
   * @return the graph containing with each vertex containing the PageRank and each edge
   *         containing the normalized weight.
   */
  def runUntilConvergence[VD: ClassTag, ED: ClassTag](
    graph: Graph[VD, ED], tol: Double, resetProb: Double = 0.15): Graph[Double, Double] =
  {
      runUntilConvergenceWithOptions(graph, tol, resetProb)
  }

  /**
   * Run a dynamic version of PageRank returning a graph with vertex attributes containing the
   * PageRank and edge attributes containing the normalized edge weight.
   *
   * @tparam VD the original vertex attribute (not used)
   * @tparam ED the original edge attribute (not used)
   *
   * @param graph the graph on which to compute PageRank
   * @param tol the tolerance allowed at convergence (smaller => more accurate).
   * @param resetProb the random reset probability (alpha)
   * @param srcId the source vertex for a Personalized Page Rank (optional)
   *
   * @return the graph containing with each vertex containing the PageRank and each edge
   *         containing the normalized weight.
   */
  def runUntilConvergenceWithOptions[VD: ClassTag, ED: ClassTag](
      graph: Graph[VD, ED], tol: Double, resetProb: Double = 0.15,
      srcId: Option[VertexId] = None): Graph[Double, Double] =
  {
    val personalized = srcId.isDefined
    val src: VertexId = srcId.getOrElse(-1L)

    // Initialize the pagerankGraph with each edge attribute
    // having weight 1/outDegree and each vertex with attribute 1.0.
    val pagerankGraph: Graph[(Double, Double), Double] = graph
      // Associate the degree with each vertex
      .outerJoinVertices(graph.outDegrees) {
        (vid, vdata, deg) => deg.getOrElse(0)
      }
      // Set the weight on the edges based on the degree
      .mapTriplets( e => 1.0 / e.srcAttr )
      // Set the vertex attributes to (initalPR, delta = 0)
      .mapVertices { (id, attr) =>
        if (id == src) (resetProb, Double.NegativeInfinity) else (0.0, 0.0)
      }
      .cache()

    // Define the three functions needed to implement PageRank in the GraphX
    // 以下将定义三个所需函数来完成GraphX对PageRank的算法实现想
    // version of Pregel
    // 第一个函数用于返回一个考虑“随机事件”发生后的计算结果
    def vertexProgram(id: VertexId, attr: (Double, Double), msgSum: Double): (Double, Double) = {
      val (oldPR, lastDelta) = attr
      val newPR = oldPR + (1.0 - resetProb) * msgSum
      (newPR, newPR - oldPR)
    }
    //
    def personalizedVertexProgram(id: VertexId, attr: (Double, Double),
      msgSum: Double): (Double, Double) = {
      val (oldPR, lastDelta) = attr
      var teleport = oldPR
      val delta = if (src==id) 1.0 else 0.0
      teleport = oldPR*delta

      val newPR = teleport + (1.0 - resetProb) * msgSum
      val newDelta = if (lastDelta == Double.NegativeInfinity) newPR else newPR - oldPR
      (newPR, newDelta)
    }
    // 第二个函数用于得到一个迭代器，里面包含了两个信息：该边的目的ID、
    // 该边的源属性值和权重的乘积（该边传递的实际PR值）
    def sendMessage(edge: EdgeTriplet[(Double, Double), Double]) = {
      if (edge.srcAttr._2 > tol) {
        Iterator((edge.dstId, edge.srcAttr._2 * edge.attr))
      } else {
        Iterator.empty
      }
    }
    // 第三个函数用于将顶点属性值和传递的值进行累加
    def messageCombiner(a: Double, b: Double): Double = a + b

    // The initial message received by all vertices in PageRank
    // 所有顶点接收到的初始信息
    val initialMessage = if (personalized) 0.0 else resetProb / (1.0 - resetProb)

    // Execute a dynamic version of Pregel.
    val vp = if (personalized) {
      (id: VertexId, attr: (Double, Double), msgSum: Double) =>
        personalizedVertexProgram(id, attr, msgSum)
    } else {
      (id: VertexId, attr: (Double, Double), msgSum: Double) =>
        vertexProgram(id, attr, msgSum)
    }

    Pregel(pagerankGraph, initialMessage, activeDirection = EdgeDirection.Out)(
      vp, sendMessage, messageCombiner)
      .mapVertices((vid, attr) => attr._1)
  } // end of deltaPageRank

}
