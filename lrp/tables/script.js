$('.tabcontent').load('table1.html');
$('.tabcontent').on('click', '#more', function() {
	console.log("yo");
	$('.advanced').toggle();
})

function openTab(url) {
	$('.tabcontent').load(url);
}

