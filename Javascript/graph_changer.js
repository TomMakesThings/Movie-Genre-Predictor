var label_graph_count = 0;
var label_graph_id = ["label-pie-raw", "genres-per-movie", "movies-per-genre"];

var desc_graph_count = 0;
var desc_graph_id = ["scattertext-horror", "scattertext-romance", "frequency-distribution", "bigrams", "trigrams"];

function graphChanger(element) {
  // Check which graphs to change
  if (element.classList.contains("label-pie")) {
    graph_count = label_graph_count
    graph_id = label_graph_id
    var current_display = document.getElementById(label_graph_id[graph_count]);
  }
  else if (element.classList.contains("label-other")) {
    graph_count = label_graph_count
    graph_id = label_graph_id
    var current_display = document.getElementById(label_graph_id[graph_count]);
  }
  else {
    graph_count = desc_graph_count
    graph_id = desc_graph_id
    var current_display = document.getElementById(desc_graph_id[graph_count]);
  }

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
  if (element.classList.contains("label-pie")) {
    label_graph_count = graph_count
    label_graph_id = graph_id
  }
  else if (element.classList.contains("label-other")) {
    label_graph_count = graph_count
    label_graph_id = graph_id
  }
  else {
    desc_graph_count = graph_count
    desc_graph_id = graph_id
  }

  // Hide the current graph and display the next
  current_display.style.display = "none";
  var next_display = document.getElementById(graph_id[graph_count]);
  next_display.style.display = "block";
}

function descriptionGraphChanger(element) {
  var current_display = document.getElementById(desc_graph_id[desc_graph_count]);
  if (element.classList.contains("next-button")) {
    desc_graph_count += 1;
  }
  else {
    desc_graph_count -= 1;
  }
  if (desc_graph_count > desc_graph_id.length - 1) {
    desc_graph_count = 0;
  }
  if (desc_graph_count < 0) {
    desc_graph_count = desc_graph_id.length - 1;
  }
  current_display.style.display = "none";
  var next_display = document.getElementById(desc_graph_id[desc_graph_count]);
  next_display.style.display = "block";
}
