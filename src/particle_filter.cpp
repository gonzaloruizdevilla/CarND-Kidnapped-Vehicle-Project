/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 40;  // TODO: Set the number of particles
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::random_device rd;
  std::default_random_engine gen(rd());
  float yaw_rate_threshold = 1.0e-3;

  for (int i = 0; i < num_particles; ++i) {
    Particle& particle = particles[i];

    double pred_x = particle.x;
    double pred_y = particle.y;
    double pred_theta = particle.theta;

    //Ignoring very low values of yaw_rate
    if (fabs(yaw_rate) < yaw_rate_threshold)
    {
      pred_x += velocity * cos(particle.theta) * delta_t;
      pred_y += velocity * sin(particle.theta) * delta_t;
    } 
    else 
    {
      pred_x += (velocity / yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
      pred_y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
      pred_theta += yaw_rate * delta_t;
    }


    
    std::normal_distribution<double> dist_x(pred_x, std_pos[0]);
    std::normal_distribution<double> dist_y(pred_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

    double new_x = dist_x(gen);
    double new_y = dist_y(gen);
    double new_theta = dist_theta(gen);

    particle.x = new_x;
    particle.y = new_y;
    particle.theta = new_theta;
  }
  
}

void ParticleFilter::dataAssociation(Particle &particle,
                                     vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations,
                                     double sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
  
  for (int i = 0; i < observations.size(); ++i) {
    int closest_id = -1;
    double lowest = sensor_range * 1.5;
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    for (int j = 0; j < predicted.size(); ++j) {
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      double current = dist(obs_x, obs_y, pred_x, pred_y);
      if (current < lowest)
      {
        lowest = current;
        closest_id = j;
      }
    }
    if (closest_id != -1){
      sense_x.push_back(obs_x);
      sense_y.push_back(obs_y);
      associations.push_back(closest_id);
    }
    observations[i].id = closest_id;
  }
  SetAssociations(particle, associations, sense_x, sense_y);
}

std::vector<LandmarkObs> ParticleFilter::transformObservations(Particle &particle,
                                                               const vector<LandmarkObs> &observations)
{
  std::vector<LandmarkObs> transformed_observations;
  double x = particle.x;
  double y = particle.y;
  double theta = particle.theta;
  for (int i = 0; i < observations.size(); ++i) {
    LandmarkObs obs = observations[i];
    double trans_x = x + obs.x * cos(theta) - obs.y * sin(theta);
    double trans_y = y + obs.x * sin(theta) + obs.y * cos(theta);
    transformed_observations.push_back(LandmarkObs{i, trans_x, trans_y});
  }
  return transformed_observations;
}

vector<LandmarkObs> ParticleFilter::inrangeLandmarks(const Particle &particle, double sensor_range, const Map &map_landmarks)
{
  vector<LandmarkObs> inrange_landmarks;
  for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {
    Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
    if ((fabs((particle.x - landmark.x_f)) <= sensor_range) &&
        (fabs((particle.y - landmark.y_f)) <= sensor_range))
    {
      inrange_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
    }
  }
  return inrange_landmarks;
}

void ParticleFilter::updateParticleWeight(Particle &particle,
                                          double std_landmark[],
                                          const vector<LandmarkObs> &observations,
                                          const vector<LandmarkObs> &landmarks)
{
  particle.weight = 1.0;
  double gauss_norm;
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  for (int i = 0; i < particle.associations.size(); ++i)
  {
    int landmark_pos = particle.associations[i];
    double land_x = landmarks[landmark_pos].x;
    double land_y = landmarks[landmark_pos].y;
    double obs_x = particle.sense_x[i];
    double obs_y = particle.sense_y[i];
    double exponent = (pow(obs_x - land_x, 2) / (2 * pow(sig_x, 2))) +
                      (pow(obs_y - land_y, 2) / (2 * pow(sig_y, 2)));

    double weight = gauss_norm * exp(-exponent);

    particle.weight *= weight;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                       const vector<LandmarkObs> &observations,
                                       const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; ++i) {
    Particle& particle = particles[i];
    
    std::vector<LandmarkObs> transformed_observations = transformObservations(particle, observations);
    
    vector<LandmarkObs> inrange_landmarks = inrangeLandmarks(particle, sensor_range, map_landmarks);
    
    dataAssociation(particle, inrange_landmarks, transformed_observations, sensor_range);

    updateParticleWeight(particle, std_landmark, transformed_observations, inrange_landmarks);

    weights[i] = particle.weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  vector<Particle> new_particles;
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::discrete_distribution<> dist {weights.begin(), weights.end()};

  for (int i = 0; i < num_particles; i++) {
    int pos = dist(gen);
    Particle particle = particles[pos];

    Particle new_particle;
    new_particle.id = i;
    new_particle.x = particle.x;
    new_particle.y = particle.y;
    new_particle.theta = particle.theta;
    new_particle.weight = particle.weight;
    new_particles.push_back(new_particle);
    weights[i] = particle.weight;
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}