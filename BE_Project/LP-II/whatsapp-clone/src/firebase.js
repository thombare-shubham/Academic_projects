// For Firebase JS SDK v7.20.0 and later, measurementId is optional
import firebase from "firebase";
const firebaseConfig = {
    apiKey: "AIzaSyB1wG4A33_NzP4aKqbnK-Aix2JT9NTT1qI",
    authDomain: "whats-app-clone-e9c35.firebaseapp.com",
    projectId: "whats-app-clone-e9c35",
    storageBucket: "whats-app-clone-e9c35.appspot.com",
    messagingSenderId: "833610870355",
    appId: "1:833610870355:web:4ff3b3d27b0e065d9720a6",
    measurementId: "G-LW0LS6GK4B"
  };

  const firebaseApp = firebase.initializeApp(firebaseConfig);
  const db = firebaseApp.firestore();
  const auth = firebase.auth();
  const provider = new firebase.auth.GoogleAuthProvider();

  export { auth , provider };
  export default db;