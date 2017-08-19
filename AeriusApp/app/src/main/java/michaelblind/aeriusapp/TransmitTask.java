package michaelblind.aeriusapp;

import android.os.AsyncTask;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

/** Created by Michael on 3/11/2017. */
class TransmitTask extends AsyncTask<Void, Void, Void> {
    private JSONObject message;
    private String  messageS;
    private boolean t = true;
    private static int vel = 0, ste = 0;
    private static int i = 0;

    private static Socket socket = null;
    private static DataOutputStream out;


    private final static String  HOST = "192.168.42.1";
    private final static int     PORT = 5015;

    TransmitTask(int v, int s) {
        if (v == vel && s == ste &&  i++ % 10 != 0) t = false;
        vel = v;
        ste = s;
        try {setMessage(v, s);}
        catch (JSONException e) {e.printStackTrace();}
    }

    TransmitTask(boolean t) {this.t = t;}

    private void setMessage(int velocity, int steering) throws JSONException {
        message = new JSONObject();
        message.put("velocity", String.format("%d", velocity));
        message.put("steering", String.format("%d", steering));
        messageS = " " + steering + "," + velocity + ".";
    }

    @Override
    protected Void doInBackground(Void... voids) {
        connectIfNec();
        if (t) transmit();
        return null;
    }

    private void connectIfNec() {
        if (socket != null) return;
        try {socket = new Socket(HOST, PORT);}
        catch (IOException e) {e.printStackTrace();}
    }

    private void transmit() {
        if (socket == null) return;
        try {
            if (out == null) out = new DataOutputStream(socket.getOutputStream());
            out.writeUTF(messageS);
            out.flush();
            print("Sent" + messageS);
        } catch (IOException e) { print(e); }
    }

    private void print(IOException e) {e.printStackTrace();}
    private void print(String string) {System.out.println(string);}

}
