

function getName(){
	
	document.getElementById('submit').disabled = false;
	document.getElementById('preview-frame').style.display = "block";
	
	var file_name = document.getElementById('file-upload').files[0].name;
	document.getElementById('fileName').innerHTML = " " + file_name;

	var reader = new FileReader();
    reader.onload = function(){
      var preview = document.getElementById('preview');
      preview.src = reader.result;
    };
    reader.readAsDataURL(event.target.files[0]);
}
