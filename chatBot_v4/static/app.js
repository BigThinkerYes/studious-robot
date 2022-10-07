class Chatbox{
    constructor(){
        this.args = {
//        type of buttons wanted
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
//        check to see if chat box open or closed
        this.state = false;
        this.messages = [];
    }
//    display messages
        display(){
            const {openButton, chatBox, sendButton} = this.args;
            openButton.addEventListener('click', () => this.toggleState(chatBox))
            sendButton.addEventListener('click', () => this.onSendButton(chatBox))

            const node = chatBox.querySelector('input');
            node.addEventListener("keyup", ({key}) =>{
                if(key === "Enter"){
                    this.onSendButton(chatBox)
                }
            })
        }

//        toggle state
        toggleState(chatbox){
            this.state = !this.state;
            // show hide
            if(this.state){
                chatbox.classList.add('chatbox--active')
            } else {
                chatbox.classList.remove('chatbox--active')
            }
        }
//        send btn
        onSendButton(chatbox){
            let textField = chatbox.querySelector('input');
            let text1 = textField.value
            if(text1 === ""){
                return;
            }

            let msg1 = { name: "User", message: text1 }
            this.messages.push(msg1);

            //http://127.0.0.1:5000/predict
            // SCRIPT_ROOT comes from html
            fetch($SCRIPT_ROOT + '/predict', {
                method: 'POST',
                body: JSON.stringify({ message: text1 }),
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                },
            })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "studious_robot", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox)
                textField.value = ''
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox)
                textField.value = ''
            });

        }
        //update
        updateChatText(chatbox){
            let html = '';
            this.messages.slice().reverse().forEach(function(item){
                if(item.name === "studious_robot")
                {
                    html += '<div class="messages__item messages__item--vistor">' + item.message + '</div>'
                }
                else
                {
                    html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
                }
            });
            const chatmessage = chatbox.querySelector('.chatbox__messages');
            chatmessage.innerHTML = html;
        }
}

const chatbox = new Chatbox();
chatbox.display(); //controls open and send eventlistener