var label_graph_count = 0;
var label_graphs_ids = ["label-pie-raw", "genres-per-movie", "movies-per-genre"];

function labelGraphChanger(element) {
  var current_display = document.getElementById(label_graphs_ids[label_graph_count]);
  if (element.classList.contains("next-button")) {
    label_graph_count += 1;
  }
  else {
    label_graph_count -= 1;
  }
  if (label_graph_count > label_graphs_ids.length - 1) {
    label_graph_count = 0;
  }
  if (label_graph_count < 0) {
    label_graph_count = label_graphs_ids.length - 1;
  }
  current_display.style.display = "none";
  var next_display = document.getElementById(label_graphs_ids[label_graph_count]);
  next_display.style.display = "inline-block";
}

var desc_graph_count = 0;
var desc_graphs_ids = ["scattertext-horror", "scattertext-romance", "frequency-distribution", "bigrams", "trigrams"];

function descriptionGraphChanger(element) {
  var current_display = document.getElementById(desc_graphs_ids[desc_graph_count]);
  if (element.classList.contains("next-button")) {
    desc_graph_count += 1;
  }
  else {
    desc_graph_count -= 1;
  }
  if (desc_graph_count > desc_graphs_ids.length - 1) {
    desc_graph_count = 0;
  }
  if (desc_graph_count < 0) {
    desc_graph_count = desc_graphs_ids.length - 1;
  }
  current_display.style.display = "none";
  var next_display = document.getElementById(desc_graphs_ids[desc_graph_count]);
  next_display.style.display = "block";
}
