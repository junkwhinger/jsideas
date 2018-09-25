---
layout: post
title:  "대한민국 인구구조 변화"
date:   2016-01-20 00:34:25
tags: [d3.js, data visualisation]

---

## 피라미드형에서 종형으로 이동하는 대한민국 인구구조
출처: <a href="http://kosis.kr/">통계청 KOSIS 총조사인구 총괄(시군구/성/5세연령별)</a>

<script src="http://d3js.org/d3.v3.min.js"></script>
<style>
    .title {
      font: 20px helvetica;
      fill: #404040;
    }

    .chart_label,
    .axis {
      font: 12px helvetica;
      fill: rgb(99,99,99);
    }

    .axis path,
    .axis line {
          color: red;
          fill: none;
          stroke: #000;
          stroke-width: 1px;
    }

    .left.bar {
      fill: #6b8891;
    }

    .right.bar {
      fill: #b27b88;
    }

    div.years_buttons {
        display: flex;
        justify-content: space-between;
        width: 1000px;
        margin-left: 37px;
      }

    div.years_buttons div {
      font: 12px helvetica;
      padding: 3px;
      margin: 7px;
      width: 55px;
      text-align: center;
    }
      
</style>
<script type="text/javascript">  
     function draw(data) {

        // setting the d3 space
        "use strict";
        var width = 960,
            height = 600;

        var margin = {
          top: 60,
          right: 60,
          bottom: 24,
          left: 60,
          middle: 28
        };
        

        // the width of each side of the chart
        var regionWidth = width/2 - margin.middle;

        // x-coordinates of the y-axes,
        // cuz they should be in the middle
        var pointA = regionWidth,
            pointB = width - regionWidth;


        // var years = [1925,1930,1935,1940,1944,1949,1955,1960,1966,1970,1975,1980,1985,1990,1995,2000,2005,2010];
        var years = [1925,1930,1935,1940,1944,1949,1955,1960,1966,1970,1975,1980,1985,1990,1995,2000,2005,2010];

        // setting the title
        // d3.select("div.chart")
        //   .append("h2")
        //   .attr("class", "title")
        //   .text("Korean Population Structure - Year 1925");

        // loading the very first batch of the population data
        var first_data = data[1925];

        // calculate the sum of the whole population of 1925
        var totalPopulation = d3.sum(first_data, function(d) { return d.male + d.female; }),
          percentage = function(d) { return d / totalPopulation; };

        // setting the svg
        var svg = d3.select('div.chart').append('svg')
          .attr("class", "container")
          .attr('width', margin.left + width + margin.right)
          .attr('height', margin.top + height + margin.bottom)
        .append('g')
          .attr('transform', translation(margin.left, margin.top));

        svg.append("text")
            .attr("class", "title")
            .attr("x", width/2)
            .attr("y", 0 - (margin.top / 2))
            .attr("text-anchor", "middle")
            .text("Korean Population Structure - Year 1925");

        // find the maximum data value on either side
        //  since this will be shared by both of the x-axes
        var maxValue = Math.max(
          d3.max(first_data, function(d) { return percentage(d.male); }),
          d3.max(first_data, function(d) { return percentage(d.female); })
        );

        // SET UP SCALES
        // the xScale goes from 0 to the width of a region
        //  it will be reversed for the left x-axis
        var xScale = d3.scale.linear()
          .domain([0, maxValue])
          .range([0, regionWidth])
          .nice();

        var xScaleLeft = d3.scale.linear()
          .domain([0, maxValue])
          .range([regionWidth, 0]);

        var xScaleLeftText = d3.scale.linear()
          .domain([0, maxValue])
          .range([0, regionWidth]);

        var xScaleRight = d3.scale.linear()
          .domain([0, maxValue])
          .range([0, regionWidth]);

        var yScale = d3.scale.ordinal()
          .domain(first_data.map(function(d) { return d.group; }))
          .rangeRoundBands([height, 0], 0.1);

        // SET UP AXES
        var yAxisLeft = d3.svg.axis()
          .scale(yScale)
          .orient('right')
          .tickSize(4,0)
          .tickPadding(margin.middle-4);

        var yAxisRight = d3.svg.axis()
          .scale(yScale)
          .orient('left')
          .tickSize(4,0)
          .tickFormat('');

        var xAxisRight = d3.svg.axis()
          .scale(xScale)
          .orient('bottom')
          .tickFormat(d3.format('.1%'));

        var xAxisLeft = d3.svg.axis()
          // REVERSE THE X-AXIS SCALE ON THE LEFT SIDE BY REVERSING THE RANGE
          .scale(xScale.copy().range([pointA, 0]))
          .orient('bottom')
          .tickFormat(d3.format('.1%'));

        // MAKE GROUPS FOR EACH SIDE OF CHART
        // scale(-1,1) is used to reverse the left side so the bars grow left instead of right
        var leftBarGroup = svg.append('g')
          .attr('transform', translation(pointA, 0) + 'scale(-1,1)');
        var rightBarGroup = svg.append('g')
          .attr('transform', translation(pointB, 0));

        // DRAW AXES
        svg.append('g')
          .attr('class', 'axis y left')
          .attr('transform', translation(pointA, 0))
          .call(yAxisLeft)
          .selectAll('text')
          .style('text-anchor', 'middle');

        svg.append('g')
          .attr('class', 'axis y right')
          .attr('transform', translation(pointB, 0))
          .call(yAxisRight);

        svg.append('g')
          .attr('class', 'axis x left')
          .attr('transform', translation(0, height))
          .call(xAxisLeft);

        svg.append('g')
          .attr('class', 'axis x right')
          .attr('transform', translation(pointB, height))
          .call(xAxisRight);

        var null_data = [{
            "group": "0-4",
            "male": 0,
            "female": 0
        }, {
            "group": "5-9",
            "male": 0,
            "female": 0

        }, {
            "group": "10-14",
            "male": 0,
            "female": 0
        }, {
            "group": "15-19",
            "male": 0,
            "female": 0
        }, {
            "group": "20-24",
            "male": 0,
            "female": 0
        }, {
            "group": "25-29",
            "male": 0,
            "female": 0
        }, {
            "group": "30-34",
            "male": 0,
            "female": 0
        }, {
            "group": "35-39",
            "male": 0,
            "female": 0
        }, {
            "group": "40-44",
            "male": 0,
            "female": 0
        }, {
            "group": "45-49",
            "male": 0,
            "female": 0
        }, {
            "group": "50-54",
            "male": 0,
            "female": 0
        }, {
            "group": "55-59",
            "male": 0,
            "female": 0
        }, {
            "group": "60-64",
            "male": 0,
            "female": 0
        }, {
            "group": "65-69",
            "male": 0,
            "female": 0
        }, {
            "group": "70-74",
            "male": 0,
            "female": 0
        }, {
            "group": "75-79",
            "male": 0,
            "female": 0
        }, {
            "group": "80-84",
            "male": 0,
            "female": 0
        }, {
            "group": "85+",
            "male": 0,
            "female": 0
        }
        ];

        leftBarGroup.selectAll('.bar.left')
          .data(null_data)
          .enter().append('rect')
            .attr('class', 'bar left')
            .attr('x', 0)
            .attr('y', function(d) { return yScale(d.group); })
            .attr('width', function(d) { return xScale(percentage(d.male)); })
            .attr('height', yScale.rangeBand());

        rightBarGroup.selectAll('.bar.right')
          .data(null_data)
          .enter().append('rect')
            .attr('class', 'bar right')
            .attr('x', 0)
            .attr('y', function(d) { return yScale(d.group); })
            .attr('width', function(d) { return xScale(percentage(d.female)); })
            .attr('height', yScale.rangeBand());

        

        leftBarGroup.selectAll('.bar.left')
          .data(first_data)
          .transition()
          .duration(1000)
          .attr('width', function(d) { return xScale(percentage(d.male)); })
          .attr('height', yScale.rangeBand());
        
        rightBarGroup.selectAll('.bar.right')
          .data(first_data)
          .transition()
          .duration(1000)
          .attr('width', function(d) { return xScale(percentage(d.female)); })
          .attr('height', yScale.rangeBand());

        var format = d3.format("0,000");

        leftBarGroup.selectAll("text")
          .data(first_data)
          .enter().append("text")
          .attr("class", "chart_label")
          .attr('transform', translation(pointA, 0) + 'scale(-1,1)')
          .attr("x", function(d) { return regionWidth - xScale(percentage(d.male)) - format(d.male).length * 7 -3; })
          .attr('y', function(d) { return yScale(d.group) + 20; })
          .text(function(d){
            return format(d.male);
          })

        rightBarGroup.selectAll("text")
          .data(first_data)
          .enter().append("text")
          .attr("class", "chart_label")
          .attr("x", function(d) { return xScale(percentage(d.female)) + format(d.male).length * 1 + 3; })
          .attr('y', function(d) { return yScale(d.group) + 20; })
          .text(function(d){
            return format(d.female);
          })


        


        // so sick of string concatenation for translations
        function translation(x,y) {
          return 'translate(' + x + ',' + y + ')';
        }

        
        function update(year) {
          // debugger;
          var filtered = data[year]
  
          d3.select(".title")
            .text("Korean Population Structure - Year " + year)

          var totalPopulation = d3.sum(filtered, function(d) { return d.male + d.female; }),
          percentage = function(d) { return d / totalPopulation; };

          var maxValue = Math.max(
          d3.max(filtered, function(d) { return percentage(d.male); }),
          d3.max(filtered, function(d) { return percentage(d.female); })
        );

          var xScale = d3.scale.linear()
          .domain([0, maxValue])
          .range([0, regionWidth])
          .nice();

          var xAxisRight = d3.svg.axis()
          .scale(xScale)
          .orient('bottom')
          .tickFormat(d3.format('.1%'));

        var xAxisLeft = d3.svg.axis()
          // REVERSE THE X-AXIS SCALE ON THE LEFT SIDE BY REVERSING THE RANGE
          .scale(xScale.copy().range([pointA, 0]))
          .orient('bottom')
          .tickFormat(d3.format('.1%'));

          svg.selectAll('.axis.x.left')
          .transition()
          .duration(500)
          .call(xAxisLeft);

        svg.selectAll('.axis.x.right')
          .transition()
          .duration(500)
          .call(xAxisRight);




          // bar update and transition
          leftBarGroup.selectAll('rect.bar.left')
            .data(filtered)
            .transition()
            .duration(500)
            .attr('width', function(d) { return xScale(percentage(d.male)); })
            .attr('height', yScale.rangeBand());

          rightBarGroup.selectAll('rect.bar.right')
            .data(filtered)
            .transition()
            .duration(500)
            .attr('width', function(d) { return xScale(percentage(d.female)); })
            .attr('height', yScale.rangeBand());

          leftBarGroup.selectAll("text")
          .data(filtered)
          .transition()
          .duration(500)
          .attr("x", function(d) { return regionWidth - xScale(percentage(d.male)) - format(d.male).length * 7 -3;; })
          .attr('y', function(d) { return yScale(d.group) + 20; })
          .text(function(d){
            return format(d.male);
          })

          rightBarGroup.selectAll("text")
          .data(filtered)
          .transition()
          .duration(500)
          .attr("x", function(d) { return xScale(percentage(d.female)) + format(d.male).length * 1 + 3; })
          .attr('y', function(d) { return yScale(d.group) + 20; })
          .text(function(d){
            return format(d.female);
          })
        }

        var year_idx = 1;

          var year_interval = setInterval(function() {
            update(years[year_idx]);

            year_idx++;

            if(year_idx >= years.length) {
                clearInterval(year_interval);
           

                var buttons = d3.select("div.chart")
                  .append("div")
                        .attr("class", "years_buttons")
                        .selectAll("div")
                        .data(years)
                        .enter()
                        .append("div")
                        .text(function(d) {
                            return d;
                        });

                buttons.on("click", function(d) {
                  d3.select(".years_buttons")
                    .selectAll("div")
                    .transition()
                    .duration(500)
                    .style("color", "black")
                    .style("background", "white");

                  d3.select(this)
                    .transition()
                    .duration(500)
                    .style("background", "black")
                    .style("color", "white");
                  update(d)
                })
            } else {
              console.log("it's okay")
            }}, 1000);
      }
</script>

<div class="chart"></div>
<script type="text/javascript">
    d3.json("/assets/materials/20160227/korea_population_sample.json", draw);
</script>
<br>

###주요 관전 포인트:
* 인구구조의 시간에 따른 변화 패턴
* 전쟁으로 인한 데이터 유실
* 출산율 저하와 노령인구 증가는 언제부터 일어나는가

###기타
* 하단 연도를 클릭해서 해당 연도의 인구구조를 불러올 수 있다.
* 그동안 깔끔하게 iframe으로 넘겨주던 bl.ocks.org가 이제 iframe 지원을 안한다고. <a href="https://github.com/mapbox/geojson.io/issues/491">mbostock님이 트래픽 땜에 고생좀 하신 듯.. </a>
* 불편하긴 하지만 jekyll 블로그에 바로 자바스크립트 코드를 박아넣는 방식을 발견하여 적용함. 차트가 2개 이상으로 넘어가면 좀 골치아파질듯.
