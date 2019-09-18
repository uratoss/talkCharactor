var socketio = io();
var myName='user'
$(function(){
    $('#message_form').submit(function(){
      msg = $('#input_msg').val()
      socketio.emit('message', msg);
      $('#messages').prepend($('<li>').text('['+myName+']>> '+ msg));
      $('#input_msg').val('');
      return false;
    });
    socketio.on('message',function(msg){
      text = msg.replace('name',myName);
      $('#messages').prepend($('<li>').text('[SEGA]>> '+text));
    });
});
