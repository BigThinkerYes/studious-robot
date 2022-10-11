window.addEventListener('mouseup', function(event){
    let chatbox2 = document.getElementById('icon');
    if(event.target != chatbox2 && event.target.parentNode !=chatbox2){
        chatbox2.style.display = 'none';
    }
});

/*window.addEventListener('mouseup', function(event){
    let chatbox2 = document.getElementById('icon');
    if(event.target != chatbox2 && event.target.parentNode !=chatbox2){
        chatbox2.style.display = 'none';
    }
});*/

/*remove multiple objects from display exampe*/
/*let baxArr = ['b1', 'b2', 'b3'];
window.addEventListener('mouseup', function(event){
    for(let i=0; i<baxArr.length; i++)
    let chatbox2 = document.getElementById('icon');
    if(event.target != chatbox2 && event.target.parentNode !=chatbox2){
        chatbox2.style.display = 'none';
    }
});*/