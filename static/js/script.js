$(document).ready(function(){ 
  if (document.getElementById('homVisible').value == "True") {
    document.getElementById('homopolymer').style.visibility = "visible";
    document.getElementById('homopolymer').style.height = "100%";
  $(this).find('[value="homopolymer"]').remove();
}
  if (document.getElementById('hairpinVisible').value == "True") {
    document.getElementById('hairpin').style.visibility = "visible";
    document.getElementById('hairpin').style.height = "100%";
  $(this).find('[value="hairpin"]').remove();
}
  if (document.getElementById('gcVisible').value == "True") {
    document.getElementById('gcmotif').style.visibility = "visible";
    document.getElementById('gcmotif').style.height = "100%";
  $(this).find('[value="gcmotif').remove();
}
});

document.getElementById('constraints').addEventListener("change", function(e) {
  var value = $(this).val();
  if (value != 'select') {
  $(this).find('[value="' +$(this).val()  +'"]').remove();
  document.getElementById(value).style.visibility = "visible";
  document.getElementById(value).style.height = "100%";
  if (value == 'homopolymer') {
    document.getElementById('homSelected').value = 'hom'
    document.getElementById('homVisible').value = "True"
  }
  if (value == 'gcmotif') {
      document.getElementById('motifGcContentSelected').value = 'gcmotif'
      document.getElementById('gcVisible').value = "True"
}
  if (value == 'hairpin') {
    document.getElementById('hairpinSelected').value = 'hairpin'
    document.getElementById('hairpinVisible').value = "True"
  }
  document.getElementById(value).trigger('load');
}
  
}, true);

function removeMotifGcContent() {
  var option = document.createElement("option");
  option.value = "gcmotif";
  option.text = "Motif GC-Content";
  document.getElementById('constraints').add(option)
  document.getElementById('gcmotif').style.visibility = "hidden";
  document.getElementById('gcmotif').style.height = 0;
  document.getElementById('motifGcContentSelected').value = ''
  document.getElementById('gcVisible').value = "False"
  document.getElementById('gcmotif').trigger('load');
}


function removeKeyGcContent() {
  var option = document.createElement("option");
  option.value = "gckey";
  option.text = "Key GC-Content";
  document.getElementById('constraints').add(option)
  document.getElementById('gckey').style.visibility = "hidden";
  document.getElementById('gckey').style.height = 0;
  document.getElementById('gckey').trigger('load');
  return;
}

function removeHomopolymer() {
  var option = document.createElement("option");
  option.value = "homopolymer";
  option.text = "Homopolymer";
  document.getElementById('constraints').add(option)
  document.getElementById('homopolymer').style.visibility = "hidden";
  document.getElementById('homopolymer').style.height = 0;
  document.getElementById('homSelected').value = ''
  document.getElementById('homVisible').value = "False"
  document.getElementById('homopolymer').trigger('load');
}

function removeHairpin() {
  var option = document.createElement("option");
  option.value = "hairpin";
  option.text = "Hairpin";
  document.getElementById('constraints').add(option)
  document.getElementById('hairpin').style.visibility = "hidden";
  document.getElementById('hairpin').style.height = 0;
  document.getElementById('hairpinSelected').value = ''
  document.getElementById('hairpinVisible').value = "False"
  document.getElementById('hairpin').trigger('load');
}
