function profileHover(element) {
  document.getElementById("tomprofile").setAttribute('src', 'Assets/Images/Profile-Watching.gif');
}

function profileUnhover(element) {
  document.getElementById("tomprofile").setAttribute('src', 'Assets/Images/Profile.png');
}

window.onscroll = function() {scrollFunction()};
window.addEventListener('resize', scrollFunction);

/* Adjust the header when scrolling down */
function scrollFunction() {
  all_content = document.getElementsByClassName('header');
  visible_header = document.getElementById('visible-header');
  var position = "absolute";
  var top = "0px";
  if (window.innerWidth <= 1300 && window.innerWidth >= 700) {
    if (document.body.scrollTop > visible_header.offsetTop || document.documentElement.scrollTop > visible_header.offsetTop) {
      position = "fixed";
      top = "-130px";
    }
  }
  // Smallest screen size
  else if (window.innerWidth <= 1300) {
    if (document.body.scrollTop > visible_header.offsetBottom || document.documentElement.scrollTop > visible_header.offsetBottom) {
      position = "fixed";
      top = "-130px";
    }
  }
  // largest screen size
  else {
    if (document.body.scrollTop > visible_header.offsetTop || document.documentElement.scrollTop > visible_header.offsetTop) {
      position = "fixed";
      top = "-80px";
    }
  }
  for (i = 0; i < all_content.length; i++) {
    all_content[i].style.position = position;
    all_content[i].style.top = top;
  }
}

/* Remove the hash from the URL when the page is reloaded to prevent page jumps */
function anchorLinks() {
  if (location.hash != '') {
      window.history.replaceState('', document.title, location.href.replace(/#.*$/, ''));
  }
}

anchorLinks();
