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
  var position = "absolute";
  var top = "0px";
  if (window.innerWidth <= 750) {
    if (document.body.scrollTop > 130 || document.documentElement.scrollTop > 130) {
      position = "fixed";
      top = "-130px";
    }
  }
  else {
    if (document.body.scrollTop > 80 || document.documentElement.scrollTop > 80) {
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
