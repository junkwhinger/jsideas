---
layout: post
title:  "2016년 테러방지법 반대 필리버스터 참여의원 현황"
date:   2016-02-28 00:34:25
img: 20160228.jpg
tags: [d3.js, data visualisation]
---

## 필리버스터 참여의원 및 발언시간 정리
출처: <a href="https://namu.wiki/w/2016%EB%85%84%20%ED%85%8C%EB%9F%AC%EB%B0%A9%EC%A7%80%EB%B2%95%20%EB%B0%98%EB%8C%80%20%ED%95%84%EB%A6%AC%EB%B2%84%EC%8A%A4%ED%84%B0/%EC%A7%84%ED%96%89%EC%83%81%ED%99%A9%20%EB%B0%8F%20%EC%B0%B8%EC%97%AC%EC%9D%98%EC%9B%90">나무위키</a>

<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3.js"></script>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous">

<style>

    .axis {
      font: 10px sans-serif;
    }

    .axis path,
    .axis line {
      fill: none;
      stroke: #000;
      shape-rendering: crispEdges;
    }

    .x.axis path {
      display: none;
    }

    .chart_title {
      font: 20px sans-serif;
    }

    .d3-tip {
      font: 10px sans-serif;
      line-height: 1.2;
      font-weight: bold;
      padding: 5px;
      background: rgba(0, 0, 0, 0.8);
      color: #fff;
      border-radius: 2px;
    }

    /* Creates a small triangle extender for the tooltip */
    .d3-tip:after {
      box-sizing: border-box;
      display: inline;
      font: 10px sans-serif;
      width: 100%;
      line-height: 1;
      color: rgba(0, 0, 0, 0.8);
      content: "\25BC";
      position: absolute;
      text-align: center;
    }

    /* Style northward tooltips differently */
    .d3-tip.n:after {
      margin: -1px 0 0 0;
      top: 100%;
      left: 0;
    }
    .button_options {
      margin-top: 10px;
      margin-left: 15px;
    }

  
</style>
<script type="text/javascript"> 
   
      function draw(data) {

        // setting the d3 space
        "use strict";
        var margin = {top:60, right:30, bottom:50, left:40};
        var width = 960 - margin.left - margin.right,
            height = 400 - margin.top - margin.bottom;

        var color_code = {'더불어민주당': '#0270b8',
                          '정의당': '#ffce00',
                          '국민의당': '#6a9e26',
                          '무소속': '#bdbdbd'
                          }

        var tip = d3.tip()
          .attr('class', 'd3-tip')
          .offset([-10, 0])
          .html(function(d) {
            var hour = Math.floor(d.duration / 3600);
            var minute = (d.duration - hour * 3600) / 60;
            var output = hour + "시간 " + minute + "분"

            return "<strong>" + d.id + "번</strong>" + "<br>" + d.mp + "<br>" + d.party + "<br>" + output;
          })
          .attr("fill", "grey");

        var formatHours = function(d) {
          var hours = Math.floor(d / 3600);
          var output = hours + 'h';
          return output;
        };

        var maxId = d3.max(data, function(d) {return d.id});
        var maxHour = d3.max(data, function(d) {return d.duration});

        var x = d3.scale.ordinal()
          .domain(data.map(function(d) {return d.mp}))
          .rangeRoundBands([0, width], .1);

        var y = d3.scale.linear()
          .domain([0, maxHour])
          .range([height, 0]);

        var xAxis = d3.svg.axis()
          .scale(x)
          .orient('bottom');

        var yAxis = d3.svg.axis()
          .scale(y)
          .orient('left')
          .tickFormat(formatHours)
          .tickValues(d3.range(0, maxHour, 3600));

        var svg = d3.select("div.chart").append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
        .append("g")
          .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        svg.append("text")
        .attr("class", "chart_title")
        .attr("x", -23)             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "left")
        .style("text-decoration", "underline")
        .text("2016년 테러방지법 반대 필리버스터 참여의원");

        svg.call(tip);

        svg.append("g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + height + ")")
          .call(xAxis)
          .selectAll("text")
          .attr("y", 5)
          .attr("x", 18)
          .attr("transform", "rotate(45)");

        svg.append("g")
          .attr("class", "y axis")
          .attr("transform", "translate(0," + "0" + ")")
          .call(yAxis);

        svg.selectAll('.bar')
           .data(data)
          .enter()
           .append("rect")
           .attr("class", "bar")
           .attr("fill", function(d) {
            return color_code[d.party];
           })
           .attr("x", function(d) {return x(d.mp)})
           .attr("width", x.rangeBand())
           .attr("y", function(d) {return y(d.duration)})
           .attr("height", function(d) { return height - y(d.duration); })
           .on('mouseover', tip.show)
           .on('mouseout', tip.hide);

        d3.select("button#speech").on('click', function () {
          var x0 = x.domain(data.sort(function(a, b){ return a.id - b.id; })
              .map(function(d) { return d.mp; }))
              .copy();

          svg.selectAll(".bar")
              .sort(function(a, b) { return x0(a.id) - x0(b.id); });

          var transition = svg.transition().duration(750),
              delay = function(d, i) { return i * 50; };

          transition.selectAll(".bar")
              .delay(delay)
              .attr("x", function(d) { return x0(d.mp); });

          transition.select(".x.axis")
              .call(xAxis)
            .selectAll("text")
              .delay(delay)
              .attr("y", 5)
                .attr("x", 18)
                .attr("transform", "rotate(45)");


        });

        d3.select("button#duration").on('click', function () {
          console.log("dfjsdf")
          var x0 = x.domain(data.sort(function(a, b){ return b.duration - a.duration; })
              .map(function(d) { return d.mp; }))
              .copy();

          svg.selectAll(".bar")
              .sort(function(a, b) { return x0(b.duration) - x0(a.duration); });

          var transition = svg.transition().duration(750),
              delay = function(d, i) { return i * 50; };

          transition.selectAll(".bar")
              .delay(delay)
              .attr("x", function(d) { return x0(d.mp); });

          transition.select(".x.axis")
              .call(xAxis)
            .selectAll("text")
              .delay(delay)
              .attr("y", 5)
                .attr("x", 18)
                .attr("transform", "rotate(45)");


        });
    }
</script>
<div class="button_options">
    <button type="button" class="btn btn-default btn-sm btn-lg" id="speech">발언순서</button>
    <button type="button" class="btn btn-default btn-sm btn-lg" id="duration">연설시간</button>
</div>
<div class="chart"></div>
<script type="text/javascript">
  d3.csv("/assets/filibuster/filibuster.csv", function(d) {
      var hour = +d.duration.split("h")[0];
      var minute = +d.duration.split("h")[1].split("m")[0];
      d.duration = hour * 3600 + minute * 60;
      return d;
    },draw);
</script>
<br>
