package svr;

import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.ujmp.core.*;



public class test {

    public static double atof(String s)
    {
        double d = Double.parseDouble(s);
        if (Double.isNaN(d) || Double.isInfinite(d))
        {
            System.err.print("NaN or Infinity in input\n");
            System.exit(1);
        }
        return(d);
    }
    //P--训练集的输入矩阵（R*Q）--100*200
    //I--训练集的输出矩阵（S*Q）--1*200
    //N--隐含层的神经元个数：Q  --200
    //IW--输入层隐含层连结权值（N*R)--200*100
    //B--隐含层神经元阈值（N*1)--200*1
    //LW--隐含层输出层连结权值（N*S）--200*1
    //function(IW,B,LW,IF,TYPE)=elm_train(P,I,N,IF,TYPE)
    public static void elm_train(double[][] P,double[] I,int N,double[][] IW,double[] B,double[] LW){
        int t = P[0].length;//200
        System.out.println("t = "+t);
        int x = P.length;//100
        System.out.println("x = "+x);
        //System.out.println("t = "+t);
        Matrix m_p=DenseMatrix.Factory.zeros(x,t);
        Matrix m_i=DenseMatrix.Factory.zeros(1,t);
        Matrix iw=DenseMatrix.Factory.randn(t,x);
        double[][] iw_temp=iw.toDoubleArray();

        for(int i=0;i<t;i++){
            for(int j=0;j<x;j++){
                IW[i][j]=iw_temp[i][j];
            }
        }
        Matrix b=DenseMatrix.Factory.rand(N,1);
        double[][] b_temp=b.toDoubleArray();
        for(int i=0;i<t;i++){
            B[i]=b_temp[i][0];
        }
        Matrix lw;
        for(int i=0;i<t;i++){
            m_i.setAsDouble(I[i],0,i);
            for(int j=0;j<x;j++){
                m_p.setAsDouble(P[j][i],j,i);
            }
        }
        Matrix temp_1=iw.mtimes(m_p);
        Matrix temp_2=DenseMatrix.Factory.zeros(t,t);
        for(int i=0;i<t;i++){
            for(int j=0;j<t;j++){
                temp_2.setAsDouble(B[j],j,i);
            }
        }
        Matrix tempH=temp_1.plus(temp_2);
        double[][] temp=tempH.toDoubleArray();
        Matrix H=DenseMatrix.Factory.zeros(t,t);
        for(int i=0;i<t;i++){
            for(int j=0;j<t;j++){
                temp[i][j]=sig(temp[i][j]);
                H.setAsDouble(temp[i][j],i,j);
            }
        }
        //System.out.println(H);
        lw=H.transpose().pinv().mtimes(m_i.transpose());
        double[][] lw_temp= lw.toDoubleArray();
        for(int i=0;i<t;i++){
            LW[i]=lw_temp[i][0];
        }
        //System.out.println(lw);
    }

    //P--训练集的输入矩阵（R*Q）--100*200
    //I--训练集的输出矩阵（S*Q）--1*200
    //N--隐含层的神经元个数：Q  --200
    //IW--输入层隐含层连结权值（N*R)--200*100
    //B--隐含层神经元阈值（N*1)--200*1
    //LW--隐含层输出层连结权值（N*S）--200*1
    //Y--计算输出层矩阵--(S*Q) --1*200
    //function Y=elm_predict(P,IW,B,LW,TYPE)
    public static void elm_predict(double[][] P,double[][] IW,double[] B,double[] LW,double[] Y){
        int t = P[0].length;//200
        int x = P.length;//100
        Matrix m_p=DenseMatrix.Factory.zeros(x,t);
        Matrix iw=DenseMatrix.Factory.zeros(t,x);
        Matrix b=DenseMatrix.Factory.zeros(t,1);
        Matrix lw=DenseMatrix.Factory.zeros(t,1);
        Matrix y;
        for(int i=0;i<t;i++){
            for(int j=0;j<x;j++) {
                m_p.setAsDouble(P[j][i], j, i);
                iw.setAsDouble(IW[i][j], i, j);
            }
            b.setAsDouble(B[i],i,0);
            lw.setAsDouble(LW[i],i,0);
        }
        Matrix temp_1=iw.mtimes(m_p);
        Matrix temp_2=DenseMatrix.Factory.zeros(t,t);
        for(int i=0;i<t;i++){
            for(int j=0;j<t;j++){
                temp_2.setAsDouble(B[j],j,i);
            }
        }
        Matrix tempH=temp_1.plus(temp_2);
        double[][] temp=tempH.toDoubleArray();
        Matrix H=DenseMatrix.Factory.zeros(t,t);
        for(int i=0;i<t;i++){
            for(int j=0;j<t;j++){
                temp[i][j]=sig(temp[i][j]);
                H.setAsDouble(temp[i][j],i,j);
            }
        }
        y=(H.transpose().mtimes(lw.transpose())).transpose();
        double[][] y_temp=y.toDoubleArray();
        for(int i=0;i<y_temp.length;i++){
            for(int j=0;j<y_temp[0].length;j++){
                System.out.println("y_temp["+i+"]["+j+"] = "+y_temp[i][j]);
            }
        }
        //System.out.println(y);
    }

    public static double sig(double t){
        return 1/(1+Math.pow(Math.E,-t));
    }

    public static void map_min_max(double[] org, double[] fina){
        int t=org.length;
        double max=map_max(org);
        double min=map_min(org);
        for(int i=0;i<t;i++){
            fina[i]=(org[i]-min)/(max-min);
        }
    }
    public static double map_max(double[] org){
        int t=org.length;
        double max=0;
        for (double v : org) {
            if (max < v) {
                max = v;
            }
        }
        return max;
    }

    public static double map_min(double[] org){
        int t=org.length;
        double min=0;
        for (double v : org) {
            if (min > v) {
                min = v;
            }
        }
        return min;
    }

    public static void main(String[] argv) throws FileNotFoundException {
        System.out.println("hello world");
//        ArrayList<Double> test= new ArrayList<>();
        int test_number=100;
        int con_number=200;
//        try {
//            BufferedReader in = new BufferedReader(new FileReader("D:\\train_data.txt"));
//            String str;
//            while ((str = in.readLine()) != null) {
////                System.out.println(str);
//                test.add(Double.valueOf(str));
//            }
//        } catch (IOException ignored) {
//        }
//        double[] P_org=new double[test_number];
//        double[] I_org=new double[test_number];
//        for(int i=0;i<test_number;i++){
//            P_org[i]=test.get(i);
//            I_org[i]=test.get(i+test_number);
//        }
//        double[] P_final=new double[test_number];
//        double[] I_final=new double[test_number];
//        map_min_max(P_org,P_final);
//        map_min_max(I_org,I_final);

//        double[] P_test=new double[test_number];
//        for(int i=0;i<test_number;i++){
//            P_test[i]=test.get(i+2*test_number);
//        }

        double[][] IW=new double[con_number][test_number];
        double[] B=new double[con_number];
        double[] LW=new double[con_number];
        double[] Yes=new double[con_number];
        double[] Y =new double[con_number];
        double[][] P_test =new double[test_number][con_number];

        double[][] P = new double[test_number][con_number];
        double[] I = new double[con_number];
        try {
            BufferedReader fp = new BufferedReader(new FileReader("D:\\train_data_final.txt"));
            int i = 0;
            while(true){
                String line = fp.readLine();
                if(line==null)break;
                StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");
                I[i]=atof(st.nextToken());
                int m = st.countTokens()/2;
                for(int j=0;j<m;j++){
                    //int temp = atoi(st.nextToken());
                    P[j][i]=atof(st.nextToken());
                }
                i++;
                //System.out.println("i : "+i);
            }
        } catch (IOException ignored) {
        }
        try{
            BufferedReader fp = new BufferedReader(new FileReader("D:\\test_data_final.txt"));
            int i=0;
            while(true){
                String line = fp.readLine();
                if(line==null)break;
                StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");
                Yes[i]=atof(st.nextToken());
                int m = st.countTokens()/2;
                for(int j=0;j<m;j++){
                    P_test[j][i]=atof(st.nextToken());
                }
                i++;
                //System.out.println("i: "+i);
            }
        }catch (IOException ignored){
        }
//        for(int i=0;i<test_number;i++){
//            System.out.println("I["+i+"] = "+I[i]);
//            for(int j=0;j<con_number;j++){
//                System.out.println("P["+i+"]["+j+"] = "+P[i][j]);
//            }
//        }

        elm_train(P, I, con_number, IW, B, LW);
        elm_predict(P_test,IW,B,LW,Y);
//        System.out.println("test_size: "+test.size());
    }
}
