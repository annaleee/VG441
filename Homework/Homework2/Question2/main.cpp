#include<iostream>
#include<fstream>

using namespace std;

const int T=52; //周期
const double D[T]={108,112,123,129,147,150,156,174,189,197,202,207,228,240,242,241,266,267,270,290,303,301,329,325,345,343,367,340,337,323,325,309,305,293,273,273,252,249,243,235,226,204,204,183,175,164,162,154,145,137,112,112};
const double s[T]={54,54,54,54,54,54,54,54,54,54,54,54};
const double h[T]={0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4};

double min(double *,int);//返回一个数组中的最小值

template<typename T>
void print(T *,int);

int main()
{
    double cost[T][T];//任意两个周期间的成本，从一个周期启动生产，库存持续到另一个周期
    double Opt_cost[T]; //记录从第一期到第T期的最优总成本
    int Order[T];  //从第一期到第T期的最优生产序列，用0-1表示，0表示该期不生产，1表示该期启动生产
    double I[T]; //记录每阶段的库存

    for (int i=0;i<T;i++)
        for (int j=0;j<T;j++)
            cost[i][j]=INT_MAX;  //初试化成本


    //计算两个周期内的成本，以及最优总成本序列
    for (int i=0;i<T;i++)
    {
        if(i>0)  //记录从第1期到第i-1期的最优总成本
        {
            double p[T];
            for (int j=0;j<T;j++)
                p[j]=cost[j][i-1];
            Opt_cost[i-1]=min(p,T);
        }

        for (int j=i;j<T;j++)
        {
            double sum=0;
            for (int k=i;k<j+1;k++)
                sum+=D[k];
            double h_sum=0;
            for (int k=i;k<j+1;k++)
            {
                h_sum=h_sum+h[i]*(sum-D[k]);
                sum=sum-D[k];
            }
            if (i>0)
                cost[i][j]=Opt_cost[i-1]+h_sum+s[i];//得到第i期到第j期的最优总成本
            else
                cost[i][j]=h_sum+s[i];
        }
    }
    double p[T];
    for (int j=0;j<T;j++)
        p[j]=cost[j][T-1];
    Opt_cost[T-1]=min(p,T);

    //求最优生产序列，从后向前推
    int i=T-1;

    while (i>=0)
    {
        if (Opt_cost[i]==cost[i][i])
        {
            Order[i]=1;
            i=i-1;
        }
        else
        {
            Order[i]=0;
            int index=i;
            for (int k=0;k<i;k++)
            {
                if (Opt_cost[i]==cost[k][i])
                {
                    index=k;
                    Order[index]=1;
                    break;
                }
            }
            Order[index]=1;
            for (int k=index+1;k<i;k++)
                Order[k]=0;
            i=index-1;
        }
    }

    //根据最优生产序列得到每个阶段的库存，从前向后推
    i=0;
    int index=0;
    while (index<T-1)
    {
        for (int j=i+1;j<T;j++)
        {
            if (Order[j]==1)
            {
                index=j;
                break;
            }
            if (j==T-1&&Order[j]==0)
            {
                index=T;
            }
        }
        double sum=0;
        for (int k=i;k<index;k++)
            sum+=D[k];
        for (int k=i;k<index;k++)
        {
            sum=sum-D[k];
            I[k]=sum;
        }
        i=index;
    }

    cout<<"最优生产序列:"<<endl;
    print(Order,T);
    //print(cost,T);
    cout<<"从第1期到各期的最优总成本:"<<endl;
    print(Opt_cost,T);
    cout<<"最优生产时的各阶段库存水平位:"<<endl;
    print(I,T);

}

double min(double *a,int n)
{
    double temp=a[0];
    for (int i=1;i<n;i++)
        if (a[i]<temp)
            temp=a[i];
    return temp;
}

template<typename T>
void print(T *a,int n)
{

    for (int i=0;i<n;i++)
        cout<<a[i]<<" ";
    cout<<endl;
}