// Counters to record which graph is currently on display
var label_count = 0;
var pie_count = 0;
var desc_count = 0;
var model_count = 0;
var ldia_count = 0;
var lsa_count = 0;

// Lists of IDs for the order of graphs to display
var label_id = ["movies-per-genre", "genres-per-movie"];
var pie_id = ["label-pie-raw", "label-pie-reduced", "label-pie-even"];
var desc_id = ["scattertext-horror", "scattertext-romance", "frequency-distribution", "bigrams", "trigrams"];
var model_id = ["model-diagram", "model-matrix", "model-pie"];
var ldia_id = ["ldia-gensim", "ldia-tsne7", "ldia-coherence", "ldia-tsne47"];
var lsa_id = ["lsa-tsne7", "lsa-coherence", "lsa-tsne2"];

function graphChanger(element) {
  // Check which graphs to change
  if (element.classList.contains("label-image")) {
    count = label_count
    id = label_id
  }
  else if (element.classList.contains("label-pie")) {
    count = pie_count
    id = pie_id
  }
  else if (element.classList.contains("desc-graphs")) {
    count = desc_count
    id = desc_id
  }
  else if (element.classList.contains("model-graphs")) {
    count = model_count
    id = model_id
  }
  else if (element.classList.contains("ldia-graphs")) {
    count = ldia_count
    id = ldia_id
  }
  else if (element.classList.contains("lsa-graphs")) {
    count = lsa_count
    id = lsa_id
  }

  var current_display = document.getElementById(id[count]);

  // Update the counter
  if (element.classList.contains("next-button")) {
    count += 1;
  }
  else {
    count -= 1;
  }
  if (count > id.length - 1) {
    count = 0;
  }
  if (count < 0) {
    count = id.length - 1;
  }

  // Update the global variable values
  if (element.classList.contains("label-image")) {
    label_count = count
    label_id = id
  }
  else if (element.classList.contains("label-pie")) {
    pie_count = count
    pie_id = id
  }
  else if (element.classList.contains("desc-graphs")) {
    desc_count = count
    desc_id = id
  }
  else if (element.classList.contains("model-graphs")) {
    model_count = count
    model_id = id
  }
  else if (element.classList.contains("ldia-graphs")) {
    ldia_count = count
    ldia_id = id
  }
  else if (element.classList.contains("lsa-graphs")) {
    lsa_count = count
    lsa_id = id
  }

  // Hide the current graph and display the next
  current_display.style.display = "none";
  var next_display = document.getElementById(id[count]);
  next_display.style.display = "block";
}
