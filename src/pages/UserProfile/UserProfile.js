import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../../contexts/UserContext';
import UserApi from '../../api/UserApi';
import styles from './UserProfile.module.css';
import { message } from 'antd';

const UserProfile = () => {
  const { user } = useUser();
  const navigate = useNavigate();
  const [isEditing, setIsEditing] = useState(false);
  const [profileData, setProfileData] = useState({
    name: '',
    email: '',
    phone: '',
    address: ''
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  useEffect(() => {
    // Check if user is logged in
    if (!user || !user.userId) {
      navigate('/login');
      return;
    }

    // Fetch user profile data
    const fetchUserProfile = async () => {
      try {
        setLoading(true);
        const response = await UserApi.getUserProfile(user.userId);
        const userData = response.data.result;
        
        setProfileData({
          name: userData.name || '',
          email: userData.email || user.email || '',
          phone: userData.phone || '',
          address: userData.address || ''
        });
        setLoading(false);
      } catch (err) {
        setError('Failed to load profile data');
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [user, navigate]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePasswordChange = (e) => {
    const { name, value } = e.target;
    setPasswordData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleEditToggle = () => {
    setIsEditing(prev => !prev);
    if (isEditing) {
      // Reset form if canceling edit
      setProfileData(prevData => ({
        ...prevData
      }));
    }
  };

  const handlePasswordToggle = () => {
    setIsChangingPassword(prev => !prev);
    if (isChangingPassword) {
      // Reset password form
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await UserApi.updateUserProfile(user.userId, profileData);
      message.success('Profile updated successfully');
      setIsEditing(false);
    } catch (err) {
      setError('Failed to update profile');
      message.error('Failed to update profile');
    }
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setError('New passwords do not match');
      message.error('New passwords do not match');
      return;
    }
    
    try {
      await UserApi.changePassword(user.userId, passwordData.currentPassword, passwordData.newPassword);
      message.success('Password changed successfully');
      setIsChangingPassword(false);
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
    } catch (err) {
      setError('Failed to change password');
      message.error('Failed to change password');
    }
  };

  if (loading) {
    return <div className={styles.loading}>Loading profile...</div>;
  }

  return (
    <div className={styles.profileContainer}>
      <h2>User Profile</h2>
      {error && <div className={styles.error}>{error}</div>}
      
      <div className={styles.profileCard}>
        <div className={styles.profileHeader}>
          <div className={styles.avatarContainer}>
            <img 
              src="https://media.istockphoto.com/id/1443209389/vi/vec-to/s%C3%A1ch-v%E1%BA%BD-ngu%E1%BB%87ch-ngo%E1%BA%A1c.jpg?s=612x612&w=0&k=20&c=a-bJw6Ic1FLS5Ri4TlRqOh0bREGyVO_2W52seuNupuk=" 
              alt="Profile" 
              className={styles.avatar} 
            />
          </div>
          <h3>{profileData.name || user.email}</h3>
        </div>

        <form onSubmit={handleSubmit} className={styles.profileForm}>
          <div className={styles.formGroup}>
            <label>Full Name</label>
            <input 
              type="text" 
              name="name" 
              value={profileData.name} 
              onChange={handleInputChange}
              disabled={!isEditing}
            />
          </div>
          
          <div className={styles.formGroup}>
            <label>Email</label>
            <input 
              type="email" 
              name="email" 
              value={profileData.email} 
              disabled={true} // Email cannot be changed
            />
          </div>
          
          <div className={styles.formGroup}>
            <label>Phone</label>
            <input 
              type="tel" 
              name="phone" 
              value={profileData.phone} 
              onChange={handleInputChange}
              disabled={!isEditing}
            />
          </div>
          
          <div className={styles.formGroup}>
            <label>Address</label>
            <textarea 
              name="address" 
              value={profileData.address} 
              onChange={handleInputChange}
              disabled={!isEditing}
              rows="3"
            />
          </div>
          
          <div className={styles.buttonGroup}>
            {isEditing ? (
              <>
                <button type="submit" className={styles.saveButton}>Save Changes</button>
                <button type="button" onClick={handleEditToggle} className={styles.cancelButton}>Cancel</button>
              </>
            ) : (
              <button type="button" onClick={handleEditToggle} className={styles.editButton}>Edit Profile</button>
            )}
          </div>
        </form>
        
        <div className={styles.passwordSection}>
          <h3>Change Password</h3>
          {isChangingPassword ? (
            <form onSubmit={handlePasswordSubmit} className={styles.passwordForm}>
              <div className={styles.formGroup}>
                <label>Current Password</label>
                <input 
                  type="password" 
                  name="currentPassword" 
                  value={passwordData.currentPassword} 
                  onChange={handlePasswordChange}
                  required
                />
              </div>
              
              <div className={styles.formGroup}>
                <label>New Password</label>
                <input 
                  type="password" 
                  name="newPassword" 
                  value={passwordData.newPassword} 
                  onChange={handlePasswordChange}
                  required
                />
              </div>
              
              <div className={styles.formGroup}>
                <label>Confirm New Password</label>
                <input 
                  type="password" 
                  name="confirmPassword" 
                  value={passwordData.confirmPassword} 
                  onChange={handlePasswordChange}
                  required
                />
              </div>
              
              <div className={styles.buttonGroup}>
                <button type="submit" className={styles.saveButton}>Change Password</button>
                <button type="button" onClick={handlePasswordToggle} className={styles.cancelButton}>Cancel</button>
              </div>
            </form>
          ) : (
            <button onClick={handlePasswordToggle} className={styles.changePasswordButton}>Change Password</button>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
