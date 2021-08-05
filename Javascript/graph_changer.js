var label_graph_count = 0;
var label_graph_id = ["movies-per-genre", "genres-per-movie"];

var pie_graph_count = 0;
var pie_graph_id = ["label-pie-raw", "label-pie-reduced", "label-pie-even"];

var desc_graph_count = 0;
var desc_graph_id = ["scattertext-horror", "scattertext-romance", "frequency-distribution", "bigrams", "trigrams"];

var model_graph_count = 0;
var model_graph_id = ["model-diagram", "model-matrix", "model-pie"];

function graphChanger(element) {
  // Check which graphs to change
  if (element.classList.contains("label-image")) {
    graph_count = label_graph_count
    graph_id = label_graph_id
  }
  else if (element.classList.contains("label-pie")) {
    graph_count = pie_graph_count
    graph_id = pie_graph_id
  }
  else if (element.classList.contains("desc-graphs")) {
    graph_count = desc_graph_count
    graph_id = desc_graph_id
  }
  else {
    graph_count = model_graph_count
    graph_id = model_graph_id
  }

  var current_display = document.getElementById(graph_id[graph_count]);

  // Update the counter
  if (element.classList.contains("next-button")) {
    graph_count += 1;
  }
  else {
    graph_count -= 1;
  }
  if (graph_count > graph_id.length - 1) {
    graph_count = 0;
  }
  if (graph_count < 0) {
    graph_count = graph_id.length - 1;
  }

  // Update the global variable values
  if (element.classList.contains("label-image")) {
    label_graph_count = graph_count
    label_graph_id = graph_id
  }
  else if (element.classList.contains("label-pie")) {
    pie_graph_count = graph_count
    pie_graph_id = graph_id
  }
  else if (element.classList.contains("desc-graphs")) {
    desc_graph_count = graph_count
    desc_graph_id = graph_id
  }
  else {
    model_graph_count = graph_count
    model_graph_id = graph_id
  }

  // Hide the current graph and display the next
  current_display.style.display = "none";
  var next_display = document.getElementById(graph_id[graph_count]);
  next_display.style.display = "block";
}
