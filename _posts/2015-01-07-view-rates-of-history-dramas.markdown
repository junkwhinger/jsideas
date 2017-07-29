---
layout:     post
title:      "지상파 사극 드라마 시청률 비교 <br> View rates of Korean History Dramas"
date:       2015-01-07 22:10:00
author:     "Jun"
categories: "d3.js"
header-img: "img/post-bg-07.jpg"
---
<h2 class="section-heading">지상파 사극 드라마 시청률 비교</h2>
<p> View Rates of Korean History dramas </p>

<div id="chart"></div>

<p> This d3 chart shows view rates of various Korean History Dramas. Click the names of dramas on the right side to compare their view rates. The x Axis is episode, and y Axis view rate.

<p> 자료 출처: Nielson Korea 시청률 (네이버 검색) </p>

<p>Img source: <a href="https://rv.wkcdn.net/http://rigvedawiki.net/r1/pds/jngdjn.jpg">Google</a></p> 

<link href="/d3_css/view_rates.css" rel='stylesheet'>
<script src="http://d3js.org/d3.v3.js"></script>
<script type="text/javascript">

//setting the canvas
var margin = {top: 20, right: 120, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

//setting the ranges
var x = d3.scale.linear()
    .range([0, width]);

var y = d3.scale.linear()
    .range([height, 0]);

//coloring lines
var color = d3.scale.category20();

//changing view rate's format to %
var formatPercent = d3.format(".0%");

//defining the axes
var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .tickFormat(formatPercent);


//defining the lines
var line = d3.svg.line()
    .interpolate("basis") // this makes the lines smoother!
    .x(function(d) { return x(d.episode); })
    .y(function(d) { return y(d.view_rate); })
    .defined(function(d) { return d.view_rate; }); // this makes the lines end where they have no data.

//creating svg
var svg = d3.select("div#chart").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//maxY is for resetting the y axis using legends
var maxY;

//function for x grid
function make_x_axis() {
  return d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .ticks(20)
}

//getting the data
d3.tsv("/d3_data/view_rates.tsv", function(error, data) {
  color.domain(d3.keys(data[0]).filter(function(key) { return key !== "episode"; }));

//changing the format from text to number
  data.forEach(function(d){
    d.episode = +d.episode;
  }) 

//organising the drama data from csv
  var dramas = color.domain().map(function(name) {
    return {
      name: name,
      values: data.map(function(d) {
        return {episode: d.episode, view_rate: +d[name]};
      }),
      // making 정도전 and 기황후 visible when the page is loaded
      visible: ((name === "정도전") | (name === "기황후") ? true : false)
    };
  });

//adding domains of x and y axis
  x.domain(d3.extent(data, function(d) { return d.episode; }));

  y.domain([
    d3.min(dramas, function(c) { return d3.min(c.values, function(v) { return v.view_rate; }); }),
    d3.max(dramas, function(c) { return d3.max(c.values, function(v) { return v.view_rate; }); })
  ]);

//adding the x Axis
  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

//adding the y Axis
  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("시청률");

//adding the paths
  var drama = svg.selectAll(".drama")
      .data(dramas)
    .enter().append("g")
      .attr("class", "drama");

  drama.append("path")
      .attr("class", "line")
      .style("pointer-events", "none")
      .attr("id", function(d) {
        return "line-" + d.name.replace(" ", "").replace("/", "");
      })
      .attr("d", function(d) { 
        return d.visible ? line(d.values) : null; })
      .style("stroke", function(d) { return color(d.name); });

  //drawing grid lines
  svg.append("g")
    .attr("class", "grid")
    .attr("transform", "translate(0," + height + ")")
    .call(make_x_axis()
      .tickSize(-height, 0, 0)
      .tickFormat("")
      )

  //drawing legend
  var legendSpace = 450 / dramas.length;
  drama.append("rect")
    .attr("width", 10)
    .attr("height", 10)
    .attr("id", function(d) {
        return "legend-" + d.name.replace(" ", "").replace("/", "");
      })
    .attr("x", width + (margin.right/3) -15)
    .attr("y", function(d,i) { return (legendSpace) + i*(legendSpace) - 8;})
    .attr("fill", function(d) {
      return d.visible ? color(d.name) : "#F1F1F2";
    })
    .attr("class", "legend-box")

    .on("click", function(d){
      d.visible = !d.visible;

      maxY = findMaxY(dramas);
      y.domain([0,maxY]);
      svg.select(".y.axis")
        .transition()
        .call(yAxis);

      drama.select("path")
        .transition()
        .attr("d", function(d) {
          return d.visible ? line(d.values) : null;
        })

      drama.select("rect")
        .transition()
        .attr("fill", function(d){
          return d.visible ? color(d.name) : "#F1F1F2";
        })
    })

//adding legend texts
  drama.append("text")
    .attr("x", width + (margin.right/3))
    .attr("y", function(d,i){ return (legendSpace) + i*(legendSpace);})
    .text(function(d) {return d.name;});



});

//adding a function to reset y axis
function findMaxY(data) {
  var maxYValues = data.map(function(d) {
    if (d.visible){
      return d3.max(d.values, function(value) {
        return value.view_rate; })
    }
  });
  return d3.max(maxYValues);
}

</script>


{% highlight javascript %}

//setting the canvas
var margin = {top: 20, right: 120, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

//setting the ranges
var x = d3.scale.linear()
    .range([0, width]);

var y = d3.scale.linear()
    .range([height, 0]);

//coloring lines
var color = d3.scale.category20();

//changing view rate's format to %
var formatPercent = d3.format(".0%");

//defining the axes
var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .tickFormat(formatPercent);


//defining the lines
var line = d3.svg.line()
    .interpolate("basis") // this makes the lines smoother!
    .x(function(d) { return x(d.episode); })
    .y(function(d) { return y(d.view_rate); })
    .defined(function(d) { return d.view_rate; }); // this makes the lines end where they have no data.

//creating svg
var svg = d3.select("div#chart").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//maxY is for resetting the y axis using legends
var maxY;

//function for x grid
function make_x_axis() {
  return d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .ticks(20)
}

//getting the data
d3.tsv("/d3_data/view_rates.tsv", function(error, data) {
  color.domain(d3.keys(data[0]).filter(function(key) { return key !== "episode"; }));

//changing the format from text to number
  data.forEach(function(d){
    d.episode = +d.episode;
  }) 

//organising the drama data from csv
  var dramas = color.domain().map(function(name) {
    return {
      name: name,
      values: data.map(function(d) {
        return {episode: d.episode, view_rate: +d[name]};
      }),
      // making 정도전 and 기황후 visible when the page is loaded
      visible: ((name === "정도전") | (name === "기황후") ? true : false)
    };
  });

//adding domains of x and y axis
  x.domain(d3.extent(data, function(d) { return d.episode; }));

  y.domain([
    d3.min(dramas, function(c) { return d3.min(c.values, function(v) { return v.view_rate; }); }),
    d3.max(dramas, function(c) { return d3.max(c.values, function(v) { return v.view_rate; }); })
  ]);

//adding the x Axis
  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

//adding the y Axis
  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("시청률");

//adding the paths
  var drama = svg.selectAll(".drama")
      .data(dramas)
    .enter().append("g")
      .attr("class", "drama");

  drama.append("path")
      .attr("class", "line")
      .style("pointer-events", "none")
      .attr("id", function(d) {
        return "line-" + d.name.replace(" ", "").replace("/", "");
      })
      .attr("d", function(d) { 
        return d.visible ? line(d.values) : null; })
      .style("stroke", function(d) { return color(d.name); });

  //drawing grid lines
  svg.append("g")
    .attr("class", "grid")
    .attr("transform", "translate(0," + height + ")")
    .call(make_x_axis()
      .tickSize(-height, 0, 0)
      .tickFormat("")
      )

  //drawing legend
  var legendSpace = 450 / dramas.length;
  drama.append("rect")
    .attr("width", 10)
    .attr("height", 10)
    .attr("id", function(d) {
        return "legend-" + d.name.replace(" ", "").replace("/", "");
      })
    .attr("x", width + (margin.right/3) -15)
    .attr("y", function(d,i) { return (legendSpace) + i*(legendSpace) - 8;})
    .attr("fill", function(d) {
      return d.visible ? color(d.name) : "#F1F1F2";
    })
    .attr("class", "legend-box")

    .on("click", function(d){
      d.visible = !d.visible;

      maxY = findMaxY(dramas);
      y.domain([0,maxY]);
      svg.select(".y.axis")
        .transition()
        .call(yAxis);

      drama.select("path")
        .transition()
        .attr("d", function(d) {
          return d.visible ? line(d.values) : null;
        })

      drama.select("rect")
        .transition()
        .attr("fill", function(d){
          return d.visible ? color(d.name) : "#F1F1F2";
        })
    })

//adding legend texts
  drama.append("text")
    .attr("x", width + (margin.right/3))
    .attr("y", function(d,i){ return (legendSpace) + i*(legendSpace);})
    .text(function(d) {return d.name;});



});

//adding a function to reset y axis
function findMaxY(data) {
  var maxYValues = data.map(function(d) {
    if (d.visible){
      return d3.max(d.values, function(value) {
        return value.view_rate; })
    }
  });
  return d3.max(maxYValues);
}

{% endhighlight %}

