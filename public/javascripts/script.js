var socketio = io();
var myName='kirasy'
$(function(){
    $('#message_form').submit(function(){
      msg = $('#input_msg').val()
      socketio.emit('message', msg);
      $('#messages').append($('<li>').text('['+myName+']>> '+ msg));
      $('#input_msg').val('');
      return false;
    });
    socketio.on('message',function(msg){
      text = msg.replace('name',myName);
      $('#messages').append($('<li>').text('[roa]>> '+text));
    });
});
