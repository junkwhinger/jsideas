---
layout:     post
title:      "Avengers 2015 - communication network"
date:       2015-05-30 01:50:00
author:     "Jun"
author:     "Jun"
tags: [d3.js, data visualisation]

---

<h2 class="section-heading">Avengers: Age of Ultron [2015] - communication network using d3.js</h2>

<p> This network graph below depicts a communication network between characters (good guys) of <a href="http://www.imdb.com/title/tt2395427/">Avengers: Age of Ultron</a>. The node size is based on the degree centrality. The thicker the link is, the more stronger the relationship is between two characters.</p>

<p> Double click a character, and you will see its immediate neighbours.</p>


<div id="chart"></div>

<link href="/assets/materials/20150530/avengers_2015.css" rel="stylesheet">
<script src="http://d3js.org/d3.v3.min.js"></script>
<script type="text/javascript">

var width = 600,
    height = 500;

var force = d3.layout.force()
    .charge(-3000)
    .linkDistance(100)
    .linkStrength(0.2)
    .size([width-50, height-50])
    .gravity(.5);

var svg = d3.select("div#chart").append("svg")
    .attr("width", width)
    .attr("height", height);


d3.json("/assets/materials/20150530/avengers_2015.json", function(error, graph) {
  force
      .nodes(graph.nodes)
      .links(graph.links)
      .start();



  var link = svg.selectAll(".link")
      .data(graph.links)
    .enter().append("line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return d.weight*3; });

  var node = svg.selectAll(".node")
      .data(graph.nodes)
    .enter().append("g")
      .attr("class", "node")
      .call(force.drag);

  node.append("circle")
      .attr("r", function(d) {return d.dc * 3})
      .style("fill", function(d) { return d.color; })
      .on('dblclick', connectedNodes); //Added code

  var text = svg.selectAll(".node")
                .append("text")
                .attr("class", "node_id")
                .attr("dx", 18)
                .attr("dy", ".35em")
                .text(function(d){ return d.id});



  node.each(collide(0.5));


  var padding = 1, // separation between circles
    radius=8;
    function collide(alpha) {
      var quadtree = d3.geom.quadtree(graph.nodes);
      return function(d) {
        var rb = 2*radius + padding,
            nx1 = d.x - rb,
            nx2 = d.x + rb,
            ny1 = d.y - rb,
            ny2 = d.y + rb;
        quadtree.visit(function(quad, x1, y1, x2, y2) {
          if (quad.point && (quad.point !== d)) {
            var x = d.x - quad.point.x,
                y = d.y - quad.point.y,
                l = Math.sqrt(x * x + y * y);
              if (l < rb) {
              l = (l - rb) / l * alpha;
              d.x -= x *= l;
              d.y -= y *= l;
              quad.point.x += x;
              quad.point.y += y;
            }
          }
          return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
        });
      };
    }


  force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    d3.selectAll("circle")
      .attr("cx", function(d) { return d.x})
      .attr("cy", function(d) { return d.y});

    d3.selectAll("text")
      .attr("x", function(d) { return d.x})
      .attr("y", function(d) { return d.y});

  });

  //Toggle stores whether the highlighting is on
  var toggle = 0;
  //Create an array logging what is connected to what
  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
      linkedByIndex[i + "," + i] = 1;
  };
  graph.links.forEach(function (d) {
      linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });
  //This function looks up whether a pair are neighbours
  function neighboring(a, b) {
      return linkedByIndex[a.index + "," + b.index];
  }
  function connectedNodes() {
      if (toggle == 0) {
          //Reduce the opacity of all but the neighbouring nodes
          d = d3.select(this).node().__data__;
          node.style("opacity", function (o) {
              return neighboring(d, o) | neighboring(o, d) ? 1 : 0.05;
          });
          link.style("opacity", function (o) {
              return d.index==o.source.index | d.index==o.target.index ? 1 : 0.05;
          });
          //Reduce the op
          toggle = 1;
      } else {
          //Put them back to opacity=1
          node.style("opacity", 1);
          link.style("opacity", 1);
          toggle = 0;
      }
  } 

});

