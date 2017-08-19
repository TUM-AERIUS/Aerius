package michaelblind.aeriusapp;

import android.os.Bundle;
import android.os.Handler;
import android.support.v4.view.MotionEventCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;

import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private int vel = 0;
    private int ste = 0;

    final int[] i = {0};

    private static final float EDGE = 200;
    private static final int DOWN = 0;
    private static final int MOVE = 2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        new TransmitTask(false).execute();
        initSendLoop();
    }

    private void initSendLoop() {
        final Handler handler = new Handler();
        Timer timer = new Timer();

        TimerTask task = new TimerTask() {
            @Override public void run() {
                handler.post(new Runnable() {
                    public void run() {new TransmitTask(vel, ste).execute(); }
                });
            }
        };

        timer.schedule(task, 700, 100);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        int action  = MotionEventCompat.getActionMasked(event);

        //Check if Event is Touch or Swipe
        if (action == DOWN || action == MOVE)
            handle(event);

        return super.onTouchEvent(event);
    }

    private void handle(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        float[] movement = transmit(x, y, getCar());
        setPointers(movement);
    }

    private void setPointers(float[] movement) {
        View[] pointers = getPointers();
        View car = getCar();

        //Start X & Y
        float sX = car.getX() + halfWidth(car) - halfWidth(pointers[0]);
        float sY = car.getY() - 0.5f * pointers[0].getHeight();

        if (movement[0] <  0) sY += car.getHeight();
        if (movement[0] == 0) sY  = car.getY();

        float step = car.getHeight() * 0.225f;

        if (movement[0] < 0) step *= -1;

        float[]   start  = new float[]{sX, sY};
        float[][] dest  = triangulateLocations(start, step, movement[1] * 1.5f);

        drawPointers(pointers, dest, movement[0]);
    }

    private float[][] triangulateLocations(float[] start, float step, float deg) {
        float microDeg = deg / 5.0f;
        float[][] loc = new float[12][2];
        loc[0] = start;

        deg = 0;

        for (int i = 1; i < 12; i++) {
            deg += microDeg;
            double rad = Math.toRadians(deg);

            float[] c = loc[i];
            c[0] = (float) (loc[i - 1][0] - step * Math.sin(rad));  //sin deg * step + loc[i-1][0]
            c[1] = (float) (loc[i - 1][1] - step * Math.cos(rad));  //cos deg * step + loc[i-1][1]
        }

        return loc;

    }

    private void drawPointers(View[] p, float[][] dest, float velocity) {
        float dec = 2.0f / (Math.abs(velocity) + 0.001f);
        for (int i = 0; i < 12; i++) {
            View v = p[i];
            float[] pos = dest[i];

            v.setX(pos[0]);
            v.setY(pos[1]);
            v.setAlpha(Math.max(0, 1.0f - i * dec));
        }

    }

    private float halfWidth(View v) { return v.getWidth() * 0.5f; }

    private View getCar() { return findViewById(R.id.car); }

    private View[] getPointers() {
        return new View[]{
                findViewById(R.id.pointer0),
                findViewById(R.id.pointer1),
                findViewById(R.id.pointer2),
                findViewById(R.id.pointer3),
                findViewById(R.id.pointer4),
                findViewById(R.id.pointer5),
                findViewById(R.id.pointer6),
                findViewById(R.id.pointer7),
                findViewById(R.id.pointer8),
                findViewById(R.id.pointer9),
                findViewById(R.id.pointer10),
                findViewById(R.id.pointer11),
                findViewById(R.id.pointer12),
                };
    }

    private float[] transmit(float x, float y, View car) {
        float y_total = car.getY() - EDGE;
        float x_total = car.getX() - EDGE;

        float y_delta = delta(y, car.getY(), car.getHeight());
        float x_delta = delta(x, car.getX(), car.getWidth());

        int velocity = (int) (absMin(y_delta, y_total) *  30f / y_total); //100f: Conversion to %
        int steering = (int) (absMin(x_delta, x_total) *  40f / x_total); // 20f: Conversion to 40°

        print (velocity, -steering);
        send  (velocity, -steering);

        return new float[]{velocity, steering};
    }

    private float delta(float z, float pos, float dim) {
        float tot = pos + dim;

        if (z < pos) return pos - z;
        if (z > tot) return tot - z;
        return 0;
    }

    /** @return out from {in; max; -max}: -max <= out <= max */
    private float absMin(float in, float max) {
        if (in >  max) return  max;
        if (in < -max) return -max;
        return in;
    }

    private void print (int v, int s) {
        TextView status = (TextView) findViewById(R.id.output);
        status.setText(v + "% " + s + "°");
    }
    private void send  (int v, int s) {
        vel = v;
        ste = s;
        /*if (!cur(v, s)) new TransmitTask(v, s).execute();*/
    }

    private boolean cur(float v, float s) {
        if ((int) v == vel && (int) s == ste) return true;
        vel = (int) v;
        ste = (int) s;
        return false;
    }
}
