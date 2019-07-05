---
layout: post
title:  "Lee Sedol vs. AlphaGo: full data"
date:   2016-03-16 12:34:25
img: 20160323.png
tags: [python, data visualisation]

---

## Thinking Time - Full Match Visualisation
Hit the buttons to navigate through the games.
Click on the circles to further investigate thinking time details.

<div class="chart"></div>

<script src="http://d3js.org/d3.v3.min.js"></script>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous">
<script src="http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3.js"></script>
  
<style>
  /*css*/

  .main_chart_title {
    margin-left: 40px;
    font: 30px helvetica;
  }

  .button_container {
    margin-left: 40px;
    margin-top: 10px;
  }

  .btn {
    margin-right: 5px;
  }

  div.chart {
  font: 10px sans-serif;
  }
  .axis path,
  .axis line {
    fill: none;
    stroke: #000;
    shape-rendering: crispEdges;
  }

  .line {
    fill: none;
    stroke-width: 1.5px;
  }

  .line#AlphaGo {
    stroke: #bf812d;
  }

  circle#AlphaGo {
    fill: #8c510a;
  }

  .line#Lee_Sedol {
    stroke: #35978f;
  }

  circle#Lee_Sedol {
    fill: #01665e;
  }

  rect#Lee_Sedol {
    fill: #01665e;
  }

  rect#AlphaGo {
    fill: #8c510a;
  }

  circle.annotation {
    fill: none;
    stroke: red;
  }

  .d3-tip {
      font: 10px sans-serif;
      line-height: 1.5;
      padding: 12px;
      background: rgba(0, 0, 0, 0.8);
      color: #fff;
      border-radius: 2px;
  }

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

    .d3-tip.n:after {
      margin: -1px 0 0 0;
      top: 100%;
      left: 0;
  }

  .ending {
    margin-left: 40px;
    font: 10px helvetica;
  }




</style>
<script type="text/javascript">
    function draw(data) {
      var games = d3.nest()
                  .key(function(d) { return d.game; })
                  .entries(who = data);

      var game1 = games[0].values
      
      var current_game = d3.max(game1, function(k) {return k.game});
    
      var margin = {top: 10, right: 60, bottom: 24, left: 60, middle: 28},
          width = 960 - margin.left - margin.right,
          height = 500 - margin.top - margin.bottom;

      var x = d3.scale.linear()
                .range([0, width]);

      var y = d3.scale.linear()
                .range([height, 0]);

      var tip = d3.tip()
          .attr('class', 'd3-tip')
          .offset([-10, 0])
          .html(function(d) {
            var time = d.thinking_time * 60;
            var minute = Math.floor(time / 60);
            var second = Math.ceil(time - minute * 60);
            var output = minute + "m " + second + "s";
            var game = d.game;
            var black_dict = {1:'Lee_Sedol', 2:'AlphaGo', 3:'Lee_Sedol', 4:'AlphaGo', 5:'Lee_Sedol'};

            var current_black = black_dict[game];

            if (d.player === current_black) {
              var ply = d.turn_index * 2 - 1;
              console.log('smae')
              return "<strong>" + d.player + "</strong>" + "<br>" + d.turn_index + "th move" +  "<br>" + ply + "th move of the game" + "<br>" + output;
            } else {
              var ply = d.turn_index * 2;
              return "<strong>" + d.player + "</strong>" + "<br>" + d.turn_index + "th move" +  "<br>" + ply + "th move of the game" + "<br>" + output;
            };

            
          })
          .attr("fill", "grey");

      var players,
          player;

      var xAxis = d3.svg.axis()
                    .scale(x)
                    .orient("bottom")
                    .ticks(20);

      var yAxis = d3.svg.axis()
                    .scale(y)
                    .orient("left");

      var line = d3.svg.line()
                   .x(function(d) { return x(d.turn_index); })
                   .y(function(d) { return y(d.thinking_time); });

      var max_turn = d3.max(game1, function(d) {return d.turn_index});
      var max_time = d3.max(game1, function(d) {return d.thinking_time});

      x.domain([0, max_turn]);
      y.domain([0, max_time]);

      var chart_title = d3.select("div.chart").append("h2")
                          .attr("class", "main_chart_title")
                          .text("Lee Sedol vs. AlphaGo: Game 1 (AlphaGo won)")

      var next_prev = ['Previous', 'Next']
      var arrow_buttons = d3.select("div.chart").append("div")
                      .attr("class", "button_container")
                      .selectAll("div")
                      .data(next_prev)
                      .enter()
                      .append("div")
                      .attr("class", "btn btn-default")
                      .attr("id", function(d){return d;})
                      .text(function(d) {
                        return d;
                      })

      var game_num = ['G1','G2','G3','G4','G5']
      var game_buttons = d3.select("div.chart").append("div")
                      .attr("class", "button_container")
                      .selectAll("div")
                      .data(game_num)
                      .enter()
                      .append("div")
                      .attr("class", "btn game_btn")
                      .attr("id", function(d) {return d;})
                      .text(function(d) {
                        return d;
                      })


      var svg = d3.select("div.chart").append("svg")
                  .attr("width", width + margin.left + margin.right)
                  .attr("height", height + margin.top + margin.bottom)
                .append("g")
                  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      svg.call(tip);

      svg.append("g")
         .attr("class", "x axis")
         .attr("transform", "translate(0," + height + ")")
         .call(xAxis)
        .append("text")
          .attr("x", width + 52)
          .attr("y", 5)
          .style("text-anchor", "end")
          .text("Turn Index");

      svg.append("g")
         .attr("class", "y axis")
         .call(yAxis)
       .append("text")
         .attr("transform", "rotate(-90)")
         .attr("y", 6)
         .attr("dy", ".71em")
         .style("text-anchor", "end")
         .text("Thinking Time in Minutes");

      var players_list = ['Lee_Sedol', 'AlphaGo'];
      
      var legend = svg.selectAll('.player')
                      .data(players_list)
                      .enter()
                      .append("g")
                      .attr("class", "player");

      legend.append("rect")
            .attr("id", function(d){return d;})
            .attr("width", 15)
            .attr("height", 15)
            .attr("x", width - 40)
            .attr("y", function(d,i) { return i*20;})
            .attr("fill", "black")

      legend.append("text")
            .attr("x", width - 20)
            .style("font-size", "12px")
            .attr("y", function(d,i) { return i*20 + 12;})
            .text(function(d) {return d;})



      players = d3.nest()
                  .key(function(d) { return d.player; })
                  .entries(game1);

      players.forEach(function(d, i) {

        
        id_var = d.key

          svg.append("path")
              .attr("id", id_var)
              .attr("class", "line")
              .attr("d", line(d.values));

        var dots = svg.selectAll("dot")
          .attr("id", id_var + "_group")
          .data(d.values)
        .enter().append("circle")
          .attr("id", id_var)
          .attr("r", 3)
          .attr("cx", function(d) { return x(d.turn_index); })
          .attr("cy", function(d) { return y(d.thinking_time); })
          .on('mouseover', tip.show)
          .on('mouseout', tip.hide);
      });

      d3.select("#G1").style("background", "black")
                        .style("color", "white")

      arrow_buttons.on("click", function(d) {
                  var direction = this.id;

                  if (direction === "Next") {
                    if (current_game !== 5) {
                      var next_game = current_game + 1;
                      console.log(current_game)
                      console.log(next_game)
                      d3.selectAll(".game_btn")
                        .transition()
                        .duration(500)
                        .style("background", "white")
                        .style("color", "black")

                      d3.select("#" + "G" + next_game)
                        .transition()
                        .duration(500)
                        .style("background", "black")
                        .style("color", "white")



                    render(next_game)

                    current_game = current_game + 1
                    }

                  } else {
                    if (current_game !== 1) {
                      var prev_game = current_game - 1;
                      console.log(current_game)
                      console.log(next_game)
                      d3.selectAll(".game_btn")
                        .transition()
                        .duration(500)
                        .style("background", "white")
                        .style("color", "black")

                      d3.select("#" + "G" + prev_game)
                        .transition()
                        .duration(500)
                        .style("background", "black")
                        .style("color", "white")



                    render(prev_game)

                    current_game = current_game - 1
                    }

                  };

      });

      game_buttons.on("click", function(d) {

                  var game_num = d.substring(1);

                      d3.selectAll(".game_btn")
                        .transition()
                        .duration(500)
                        .style("background", "white")
                        .style("color", "black")

                      d3.select("#" + d)
                        .transition()
                        .duration(500)
                        .style("background", "black")
                        .style("color", "white")



                    render(game_num)


      });

      function render(game_number) {

        var this_game = games[game_number - 1].values

        var current_game = d3.max(this_game, function(k) {return k.game});

        var winner_dict = {1:'AlphaGo', 2:'AlphaGo', 3:'AlphaGo', 4:'Lee_Sedol', 5:'AlphaGo'}

        var current_winner = winner_dict[game_number]

        var max_turn = d3.max(this_game, function(d) {return d.turn_index});
        var max_time = d3.max(this_game, function(d) {return d.thinking_time});

        x.domain([0, max_turn]);
        y.domain([0, max_time]);

        if (game_number == 4) {
          d3.select(".main_chart_title")
          .text("Lee Sedol vs. AlphaGo: Game " + current_game + " (" + current_winner + " won! yeeasssssh!)");

          svg.append("circle")
            .attr("class", "annotation")
            .attr("id", "g4")
            .attr("r", 5)
            .attr("cx", 364)
            .attr("cy", 276);

          svg.append("text")
            .attr("class", "annotation")
            .attr("id", "g4")
            .attr("x", 368)
            .attr("y", 272)
            .text("Lee Sedol's brilliant attack")

          svg.append("circle")
            .attr("class", "annotation")
            .attr("id", "g4")
            .attr("r", 5)
            .attr("cx", 410)
            .attr("cy", 415);

          svg.append("text")
            .attr("class", "annotation")
            .attr("id", "g4")
            .attr("x", 410)
            .attr("y", 400)
            .text("Consistent mistakes by AlphaGo")

        } else {

          svg.selectAll(".annotation").remove()
          d3.select(".main_chart_title")
          .text("Lee Sedol vs. AlphaGo: Game " + current_game + " (" + current_winner + " won)");
        }

        
        players = d3.nest()
                  .key(function(d) { return d.player; })
                  .entries(this_game);

        players.forEach(function(d, i) {

          id_var = d.key

          d3.select(".line#" + id_var)
            .transition()
            .duration(500)
            .attr("d", line(d.values));

          var n_dots = svg.selectAll('circle#' + id_var)
                          .data(d.values)

          n_dots.transition()
            .duration(500)
            .attr("cx", function(d) { return x(d.turn_index); })
            .attr("cy", function(d) { return y(d.thinking_time); });

          n_dots.enter().append("circle")
            .attr("id", id_var)
            .attr("r", 3)
            .attr("cx", function(d) { return x(d.turn_index); })
            .attr("cy", function(d) { return y(d.thinking_time); });

          n_dots.exit().remove();

          svg.selectAll(".axis.x")
             .transition()
             .duration(500)
             .call(xAxis);

          svg.selectAll(".axis.y")
             .transition()
             .duration(500)
             .call(yAxis);

      });
    }

     
    }
</script>
<p class="ending">made by <a target="_blank" href="http://jsideas.net">junkwhinger</a></p>

<script type="text/javascript">

  d3.csv("/assets/materials/20160316//games.csv", function(d){
      d.turn_index = +d.turn_index;
      d.thinking_time = +d.thinking_time / 60;
      d.game = +d.game;
      return d;
    }, draw)

</script>
<hr>

## A few more data visualisations with python
<br>
### Game 1
![Game 1](/assets/materials/20160316//g1_edited.png)

### Game 2
![Game 2](/assets/materials/20160316//g2_edited.png)

### Game 3
![Game 3](/assets/materials/20160316//g3_edited.png)

### Game 4
![Game 4](/assets/materials/20160316//g4_edited.png)

### Game 5
![Game 5](/assets/materials/20160316//g5_edited.png)

### Thinking Time Distribution
![KDE plot](/assets/materials/20160316//KDE_total.png)

### Thinking Time Distribution: Lee Sedol
![KDE plot](/assets/materials/20160316//KDE_lee.png)

### Thinking Time Distribution: AlphaGo
![KDE plot](/assets/materials/20160316//KDE_alpha.png)

### Thinking Time Remaining
![Point plot](/assets/materials/20160316//total_remaining_edited.png)

### Thinking Time Remaining: Each Game
![KDE plot](/assets/materials/20160316//separate_remaining.png)

<hr>

## Raw Data
full data available on <a target="_blank" href="https://github.com/junkwhinger/AlphaGo_raw">GitHub</a>.