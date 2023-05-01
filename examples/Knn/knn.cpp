#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
//#include "linalg.h"

using namespace std;

//vector<int> knn(vector<vector<int>> data, vector<vector<int>> predict, int k=3){
//  vector<int> distances;
//  for(int i=0; i<data.size(); i++){
//    for (int j=0; j<data[i].size(); j++){
//      int euclidean_distance = linalg::norm(data[i][j]-predict);
//      distances.push_back([euclidean_distance,data[i]]):
//    }
//  }
//  //votes = [i[1] for i in sorted(distances)[:k]]
//  //print(votes)
//  //vote_result = Counter(votes).most_common(1)[0][0]
//  return distances;
//}

vector<vector<string>> read_csv(string path){
  ifstream file(path);
  vector<vector<string>> data;

  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    vector<string> row;
    string cell;

    while (getline(ss, cell, ',')) {
        row.push_back(cell);
    }

    data.push_back(row);
  }
  return data;
}

int main() {

  vector<vector<string>> data = read_csv("./data/DATA.csv");
  for(int i=0; i<data.size(); i--){
    for(int i=0; j<data[i].size();j++){
      cout << data[i][j] << " ";
    }
    cout << endl;
  }

}
