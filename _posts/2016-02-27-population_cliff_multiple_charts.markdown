---
layout: post
title:  "Population Structure Change - Korea and Japan"
date:   2016-02-27 00:34:25
categories: d3
image: /assets/population_cliff/header.png
---

## Population Structure change over time - Korea and Japan
data source: <a href="http://kosis.kr/">Statistics Korea</a>, <a href="http://www.stat.go.jp/english/">Statistics Japan</a>

<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="https://d3js.org/d3.v3.min.js" charset="utf-8"></script>
<style>
.title {
    font: 20px helvetica;
    fill: #404040;
    text-transform: capitalize;
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
  .left.bar,
  .legend_male {
    fill: #6b8891;
  }
  .right.bar,
  .legend_female {
    fill: #b27b88;
  }

  .legend_text {
    font: 13px helvetica;
  }

  div.years_buttons {
      display: flex;
      justify-content: space-between;
      width: 900px;
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

  
    // setting the margin and space
    var margin = {top: 60, right: 30, bottom: 24, left: 30, middle: 28},
        width = 450 - margin.left - margin.right,
        height = 450 - margin.top - margin.bottom;

    var regionWidth = width/2 - margin.middle;

    var pointA = regionWidth,
        pointB = width - regionWidth;

    var formatPercent = d3.format(".0%");

    // simple two levels: 1) country 2) year

    var year_data = d3.nest()
                      .key(function(d) {return d.year;})
                      .key(function(d) {return d.country})
                      .entries(data);

    // to reverse the axes for female
    function unique(x) {
      return x.reverse().filter(function (e, i, x) {return x.indexOf(e, i+1) === -1;}).reverse();
    }

    // to make translate easier
    function translation(x,y) {
        return 'translate(' + x + ',' + y + ')';
    }


      // use the very first year data for the chart
      var countries = year_data[0].values;
    
      var i = 0;
      // plot charts for each country
      countries.forEach(function(c) {

        c.totalPopulation = d3.sum(c.values, function(d) {return d.population; });
        var percentage = function(d) {return d/ c.totalPopulation;};

        array_by_age = d3.nest()
                 .key(function(d) {return d.age_bin;})
                 .entries(c.values)

        // find the max x value to set the scale of the plots
        c.maxValue = Math.max(
          d3.max(array_by_age, function(d) {
            var male_val = d.values.filter(function(e) {return e.sex === 'male'})[0].population;
            return percentage(male_val);
          }),
          d3.max(array_by_age, function(d) {
            var female_val = d.values.filter(function(e) {return e.sex === 'female'})[0].population;
            return percentage(female_val);
          }) 
        );

        // set svg
        var svg = d3.select("div.chart").append('svg')
          .attr("class", "population_chart")
          .attr("id", c.key)
          .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
              .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // add titles
        svg.append("text")
           .attr("class", "title")
           .attr("x", width / 2)
           .attr("y", 0 - (margin.top / 3))
           .attr("text-anchor", "middle")
           .text(function(d) {
             return c.key + "-" + c.values[0].year;
           });

        // draw legend
        if (c.key == 'korea') {

          svg.append("rect")
             .attr("class", "legend_male")
             .attr("x", 0)
             .attr("y", 0)
             .attr("width", 15)
             .attr("height", 15);

          svg.append("rect")
             .attr("class", "legend_female")
             .attr("x", 0)
             .attr("y", 20)
             .attr("width", 15)
             .attr("height", 15);

          svg.append("text")
             .attr("class", "legend_text")
             .attr("x", 18)
             .attr("y", 12)
             .text("male");

          svg.append("text")
             .attr("class", "legend_text")
             .attr("x", 18)
             .attr("y", 32)
             .text("female");
        }
        
        // set xScale
        var xScale = d3.scale.linear()
           .domain([0, c.maxValue])
           .range([0, regionWidth])
           .nice();
        
        // set xScale for male
        var xScaleLeft = d3.scale.linear()
           .domain([0, c.maxValue])
           .range([regionWidth, 0]);

        // set xScale for female
        var xScaleRight = d3.scale.linear()
           .domain([0, c.maxValue])
           .range([0, regionWidth]);

        // set yScale
        var yScale = d3.scale.ordinal()
           .domain(unique(c.values.map(function(d) {return d.age_bin})))
           .rangeRoundBands([height, 0], 0.1);

        // set yAxis for male
        var yAxisLeft = d3.svg.axis()
           .scale(yScale)
           .orient('right')
           .tickSize(4,0)
           .tickPadding(margin.middle - 4);

        // set yAxis for female
        var yAxisRight = d3.svg.axis()
           .scale(yScale)
           .orient('left')
           .tickSize(4,0)
           .tickFormat('');

        // set xAxis for female
        var xAxisRight = d3.svg.axis()
           .scale(xScale)
           .orient('bottom')
           .tickFormat(d3.format('%'))
           .ticks(5);

        // set xAxis for male
        var xAxisLeft = d3.svg.axis()
           .scale(xScale.copy().range([pointA, 0]))
           .orient('bottom')
           .tickFormat(d3.format('%'))
           .ticks(5);

        var leftBarGroup = svg.append('g')
           .attr('class', 'lbg')
           .attr('transform', translation(pointA, 0) + 'scale(-1,1)');

        var rightBarGroup = svg.append('g')
           .attr('class', 'rbg')
           .attr('transform', translation(pointB, 0));

        // left y axis group
        svg.append('g')
           .attr('class', 'axis y left')
           .attr('transform', translation(pointA, 0))
           .call(yAxisLeft)
           .selectAll('text')
           .style('text-anchor', 'middle');

        // right y axis group
        svg.append('g')
           .attr('class', 'axis y right')
           .attr('transform', translation(pointB, 0))
           .call(yAxisRight);

        // left x axis group
        svg.append('g')
           .attr('class', 'axis x left')
           .attr('transform', translation(0, height))
           .call(xAxisLeft);

        // right x axis group
        svg.append('g')
          .attr('class', 'axis x right')
          .attr('transform', translation(pointB, height))
          .call(xAxisRight);

        // data that contains 0 for nice animation when the page starts
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
           }];

        // initiate the leftBarGroup
        leftBarGroup.selectAll('.bar.left')
          .data(null_data)
          .enter().append('rect')
            .attr('class', 'bar left')
            .attr('x', 0)
            .attr('y', function(d) { return yScale(d.group); })
            .attr('width', function(d) { return xScale(percentage(d.male)); })
            .attr('height', yScale.rangeBand());

        // initiate the rightBarGroup
        rightBarGroup.selectAll('.bar.right')
          .data(null_data)
          .enter().append('rect')
            .attr('class', 'bar right')
            .attr('x', 0)
            .attr('y', function(d) { return yScale(d.group); })
            .attr('width', function(d) { return xScale(percentage(d.female)); })
            .attr('height', yScale.rangeBand());

        
        var c_data = d3.nest().key(function(d){return d.sex}).entries(c.values);
        var male_data = c_data[0].values;
        var female_data = c_data[1].values;

        // update the data for male
        leftBarGroup.selectAll('.bar.left')
          .data(male_data)
          .transition()
          .duration(1000)
          .attr('width', function(d) {
            return xScale(percentage(d.population)); })
          .attr('height', yScale.rangeBand());
        
        // update the data for female
        rightBarGroup.selectAll('.bar.right')
          .data(female_data)
          .transition()
          .duration(1000)
          .attr('width', function(d) { return xScale(percentage(d.population)); })
          .attr('height', yScale.rangeBand());

          // to the next year group
          i += 1;


      });

      
    // update function
    // bring new year_data
    // find new max value for axis scaling
    // reset axes and update data
    function update(year_idx) {

      var countries = year_data[year_idx].values;

      countries.forEach(function(c) {

        var target_country = c.key;

        c.totalPopulation = d3.sum(c.values, function(d) {return d.population; });
        var percentage = function(d) {return d/ c.totalPopulation;};

        array_by_age = d3.nest()
                 .key(function(d) {return d.age_bin;})
                 // .key(function(d) {return d.sex})
                 .entries(c.values)

        c.maxValue = Math.max(
          d3.max(array_by_age, function(d) {
            var male_val = d.values.filter(function(e) {return e.sex === 'male'})[0].population;
            return percentage(male_val);
          }),
          d3.max(array_by_age, function(d) {
            var female_val = d.values.filter(function(e) {return e.sex === 'female'})[0].population;
            return percentage(female_val);
          }) 
        );

        var xScale = d3.scale.linear()
        .domain([0, c.maxValue])
        .range([0, regionWidth])
        .nice();

        var yScale = d3.scale.ordinal()
           .domain(unique(c.values.map(function(d) {return d.age_bin})))
           .rangeRoundBands([height, 0], 0.1);

        var xAxisRight = d3.svg.axis()
           .scale(xScale)
           .orient('bottom')
           .tickFormat(d3.format('%'))
           .ticks(5);

        var xAxisLeft = d3.svg.axis()
           .scale(xScale.copy().range([pointA, 0]))
           .orient('bottom')
           .tickFormat(d3.format('%'))
           .ticks(5);

        var n_svg = d3.select("svg#" +c.key)
          .data([c.values])
      
        n_svg.select(".title")
        .transition()
        .duration(1000)
        .text(function(c) {
          return c[0].country + "-" + c[0].year
           });



        n_svg.selectAll('.axis.x.left')
          .transition()
          .duration(1000)
          .call(xAxisLeft);

        n_svg.selectAll('.axis.x.right')
          .transition()
          .duration(1000)
          .call(xAxisRight);


        var c_data = d3.nest().key(function(d){return d.sex}).entries(c.values);
        var male_data = c_data[0].values;
        var female_data = c_data[1].values;


        n_svg.select(".lbg").selectAll('rect.bar.left')
          .data(male_data)
          .transition()
          .duration(1000)
          .attr('width', function(d) { return xScale(percentage(d.population)); })
          .attr('height', yScale.rangeBand());

        n_svg.select(".rbg").selectAll('rect.bar.right')
          .data(female_data)
          .transition()
          .duration(1000)
          .attr('width', function(d) { return xScale(percentage(d.population)); })
          .attr('height', yScale.rangeBand());

      });


    }

    var years = [];

    var year_idx = 1;

    year_data.forEach(function(y) {
      years.push(y.key)
    })

    var year_interval = setInterval(function() {
          update([year_idx]);

          year_idx++;
      
          // if the year index exceeds the data length,
          // stop the interval
          if(year_idx >= years.length ) {

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
                update(years.indexOf(d))
              })
         
          } else {
            // auto-update debugging
            // console.log("it's okay")
          }}, 1000);
  }  
</script>

<div class="chart"></div>
<script type="text/javascript">
  d3.csv("/assets/population_cliff/population.csv", function(d){
      d.population = +d.population;
      return d;
    }, draw);
</script>
<br>
