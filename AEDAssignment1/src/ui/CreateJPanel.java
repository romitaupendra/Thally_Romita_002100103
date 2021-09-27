/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JPanel.java to edit this template
 */
package ui;

import java.awt.Image;
import java.io.File;
import java.util.Date;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.filechooser.FileNameExtensionFilter;
import model.User;

/**
 *
 * @author 17814
 */
public class CreateJPanel extends javax.swing.JPanel {

    /**
     * Creates new form CreateJPanel
     */
    User user;
    public CreateJPanel(User user) {
        initComponents();
        this.user = user;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        lblCreateUser = new javax.swing.JLabel();
        lblFullName = new javax.swing.JLabel();
        lblAddress = new javax.swing.JLabel();
        lblBirthDate = new javax.swing.JLabel();
        lblMobileNum = new javax.swing.JLabel();
        txtFName = new javax.swing.JTextField();
        lblFaxNum = new javax.swing.JLabel();
        lblEmailAddress = new javax.swing.JLabel();
        lblSocSecNum = new javax.swing.JLabel();
        lblMedRecNum = new javax.swing.JLabel();
        lblHealthPlanNum = new javax.swing.JLabel();
        lblBankAccNum = new javax.swing.JLabel();
        lblLicenseNum = new javax.swing.JLabel();
        lblVehType = new javax.swing.JLabel();
        txtBirthDate = new javax.swing.JTextField();
        txtMobileNum = new javax.swing.JTextField();
        txtFaxNum = new javax.swing.JTextField();
        txtEmailAddress = new javax.swing.JTextField();
        txtSocSecNum = new javax.swing.JTextField();
        txtMedRecNum = new javax.swing.JTextField();
        txtHealthPlanNum = new javax.swing.JTextField();
        txtBankAccNum = new javax.swing.JTextField();
        txtLicenseNum = new javax.swing.JTextField();
        lblVehNumber = new javax.swing.JLabel();
        lblDeviceId = new javax.swing.JLabel();
        lblLinkedIn = new javax.swing.JLabel();
        lblBioId = new javax.swing.JLabel();
        txtVehNum = new javax.swing.JTextField();
        txtDeviceId = new javax.swing.JTextField();
        txtLinkedIn = new javax.swing.JTextField();
        jScrollPane1 = new javax.swing.JScrollPane();
        txtAddress = new javax.swing.JTextArea();
        ddVehType = new javax.swing.JComboBox<>();
        btnSubmitUserInfo = new javax.swing.JToggleButton();
        lblIpAddress = new javax.swing.JLabel();
        txtIpAddress = new javax.swing.JTextField();
        btnUploadPhoto = new javax.swing.JButton();
        lblUploadPic = new javax.swing.JLabel();

        lblCreateUser.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
        lblCreateUser.setHorizontalAlignment(javax.swing.SwingConstants.LEFT);
        lblCreateUser.setText("                                    Create User Profile");

        lblFullName.setText("Full Name");

        lblAddress.setText("Mailing Address");

        lblBirthDate.setText("Date of Birth");

        lblMobileNum.setText("Mobile Number");

        txtFName.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtFNameActionPerformed(evt);
            }
        });

        lblFaxNum.setText("Fax Number");

        lblEmailAddress.setText("Email Address");

        lblSocSecNum.setText("Social Security Number");

        lblMedRecNum.setText("Medical Record Number");

        lblHealthPlanNum.setText("Health Plan Beneficiary Number");

        lblBankAccNum.setText("Bank Account Number");

        lblLicenseNum.setText("License Number");

        lblVehType.setText("Vehicle Type");

        txtMobileNum.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtMobileNumActionPerformed(evt);
            }
        });

        txtFaxNum.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtFaxNumActionPerformed(evt);
            }
        });

        txtSocSecNum.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtSocSecNumActionPerformed(evt);
            }
        });

        txtHealthPlanNum.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtHealthPlanNumActionPerformed(evt);
            }
        });

        lblVehNumber.setText("Vehicle Number");

        lblDeviceId.setText("Device Identifier");

        lblLinkedIn.setText("LinkedIn");

        lblBioId.setText("Biometric Identifier/Photo");

        txtDeviceId.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtDeviceIdActionPerformed(evt);
            }
        });

        txtAddress.setColumns(20);
        txtAddress.setRows(5);
        jScrollPane1.setViewportView(txtAddress);

        ddVehType.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Two wheeler", "Four wheeler" }));

        btnSubmitUserInfo.setText("Submit");
        btnSubmitUserInfo.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnSubmitUserInfoActionPerformed(evt);
            }
        });

        lblIpAddress.setText("IP Address");

        txtIpAddress.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtIpAddressActionPerformed(evt);
            }
        });

        btnUploadPhoto.setText("Upload");
        btnUploadPhoto.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnUploadPhotoActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(83, 83, 83)
                .addComponent(lblBioId)
                .addGap(10, 10, 10)
                .addComponent(lblUploadPic, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnUploadPhoto)
                .addGap(246, 246, 246))
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(161, 161, 161)
                        .addComponent(lblFullName)
                        .addGap(10, 10, 10)
                        .addComponent(txtFName, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(175, 175, 175)
                        .addComponent(lblBirthDate))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(132, 132, 132)
                        .addComponent(lblAddress)
                        .addGap(10, 10, 10)
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(60, 60, 60)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblMobileNum)
                            .addGroup(layout.createSequentialGroup()
                                .addGap(15, 15, 15)
                                .addComponent(lblFaxNum))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(3, 3, 3)
                                .addComponent(lblEmailAddress))))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(97, 97, 97)
                        .addComponent(lblSocSecNum)
                        .addGap(10, 10, 10)
                        .addComponent(txtSocSecNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(159, 159, 159)
                        .addComponent(lblLicenseNum))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(93, 93, 93)
                        .addComponent(lblMedRecNum)
                        .addGap(10, 10, 10)
                        .addComponent(txtMedRecNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(197, 197, 197)
                        .addComponent(lblLinkedIn))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(55, 55, 55)
                        .addComponent(lblHealthPlanNum)
                        .addGap(10, 10, 10)
                        .addComponent(txtHealthPlanNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(185, 185, 185)
                        .addComponent(lblIpAddress)))
                .addGap(2, 2, 2)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(txtLicenseNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtLinkedIn, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtBirthDate, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtMobileNum, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtFaxNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtEmailAddress, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtIpAddress, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 0, Short.MAX_VALUE))
            .addGroup(layout.createSequentialGroup()
                .addGap(522, 522, 522)
                .addComponent(btnSubmitUserInfo))
            .addGroup(layout.createSequentialGroup()
                .addGap(129, 129, 129)
                .addComponent(lblDeviceId)
                .addGap(10, 10, 10)
                .addComponent(txtDeviceId, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
            .addGroup(layout.createSequentialGroup()
                .addGap(133, 133, 133)
                .addComponent(lblVehNumber)
                .addGap(10, 10, 10)
                .addComponent(txtVehNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
            .addGroup(layout.createSequentialGroup()
                .addGap(147, 147, 147)
                .addComponent(lblVehType)
                .addGap(10, 10, 10)
                .addComponent(ddVehType, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
            .addGroup(layout.createSequentialGroup()
                .addGap(101, 101, 101)
                .addComponent(lblBankAccNum)
                .addGap(10, 10, 10)
                .addComponent(txtBankAccNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
            .addComponent(lblCreateUser, javax.swing.GroupLayout.PREFERRED_SIZE, 639, javax.swing.GroupLayout.PREFERRED_SIZE)
        );

        layout.linkSize(javax.swing.SwingConstants.HORIZONTAL, new java.awt.Component[] {btnUploadPhoto, txtBankAccNum, txtBirthDate, txtDeviceId, txtEmailAddress, txtFName, txtFaxNum, txtHealthPlanNum, txtIpAddress, txtLicenseNum, txtLinkedIn, txtMedRecNum, txtMobileNum, txtSocSecNum, txtVehNum});

        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(11, 11, 11)
                .addComponent(lblCreateUser)
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(txtFName, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtBirthDate, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblFullName)
                            .addComponent(lblBirthDate))))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(lblAddress)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 64, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addComponent(lblMobileNum)
                        .addGap(12, 12, 12)
                        .addComponent(lblFaxNum)
                        .addGap(12, 12, 12)
                        .addComponent(lblEmailAddress))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(txtMobileNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(6, 6, 6)
                        .addComponent(txtFaxNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(6, 6, 6)
                        .addComponent(txtEmailAddress, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addGap(3, 3, 3)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(txtSocSecNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtLicenseNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblSocSecNum)
                            .addComponent(lblLicenseNum))))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(txtMedRecNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(txtLinkedIn, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblMedRecNum)
                            .addComponent(lblLinkedIn))))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(lblHealthPlanNum)
                    .addComponent(txtHealthPlanNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addComponent(lblIpAddress))
                    .addComponent(txtIpAddress, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(lblBankAccNum)
                    .addComponent(txtBankAccNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addComponent(lblVehType))
                    .addComponent(ddVehType, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addComponent(lblVehNumber))
                    .addComponent(txtVehNum, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(3, 3, 3)
                        .addComponent(lblDeviceId))
                    .addComponent(txtDeviceId, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(0, 0, 0)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblBioId)
                    .addComponent(lblUploadPic, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnUploadPhoto))
                .addGap(10, 10, 10)
                .addComponent(btnSubmitUserInfo))
        );

        layout.linkSize(javax.swing.SwingConstants.VERTICAL, new java.awt.Component[] {btnUploadPhoto, txtBankAccNum, txtBirthDate, txtDeviceId, txtEmailAddress, txtFName, txtFaxNum, txtHealthPlanNum, txtIpAddress, txtLicenseNum, txtLinkedIn, txtMedRecNum, txtMobileNum, txtSocSecNum, txtVehNum});

    }// </editor-fold>//GEN-END:initComponents
    
    private boolean validField(JTextField fieldName, String errorMsg) {
        if (fieldName.getText().trim().length() == 0) {
            JOptionPane.showMessageDialog(this, "Please complete the form");
            return false;
        }
        return true;
    }
    private void btnSubmitUserInfoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnSubmitUserInfoActionPerformed
        // TODO add your handling code here:
        
        int socSecNumber;
        int medRecNumber;
        int healthPlanNum;
        int bankAccNum;
        int mobNum;
        int faxNum;
        int licenseNum;
       
        if (validField(txtFName, TOOL_TIP_TEXT_KEY)) {
            try {
                socSecNumber = Integer.parseInt(txtSocSecNum.getText());
                medRecNumber = Integer.parseInt(txtMedRecNum.getText());
                healthPlanNum = Integer.parseInt(txtHealthPlanNum.getText());
                bankAccNum = Integer.parseInt(txtBankAccNum.getText());
                mobNum = Integer.parseInt(txtMobileNum.getText());
                faxNum = Integer.parseInt(txtFaxNum.getText());
                licenseNum = Integer.parseInt(txtLicenseNum.getText());
                
            } catch (NumberFormatException e) {
                JOptionPane.showMessageDialog(this, "Invalid data", "Info", JOptionPane.INFORMATION_MESSAGE);
                return;
            }
            if (!(txtFName.getText().matches("^[a-zA-Z]*$"))) {
                JOptionPane.showMessageDialog(this, "Please enter valid name. Name can only contain alphabets", "Info", JOptionPane.INFORMATION_MESSAGE);
                return;
            
            }
        
        user.setfName(txtFName.getText());
        user.setMailAddress(txtAddress.getText());
        user.setbDate(txtBirthDate.getText());
        user.setMobNum(txtMobileNum.getText());
        user.setFaxNum(txtFaxNum.getText());
        user.setEmailId(txtEmailAddress.getText());
        user.setSocSecNum(txtSocSecNum.getText());
        user.setMedRecNum(txtMedRecNum.getText());
        user.setHealthPlanBenNum(txtHealthPlanNum.getText());
        user.setBanAccNum(txtBankAccNum.getText());
        user.setCertLicNum(txtLicenseNum.getText());
        user.setVehType(ddVehType.getSelectedItem().toString());
        user.setVehNumber(txtVehNum.getText());
        user.setDevIdNum(txtDeviceId.getText());
        user.setLinkedinId(txtLinkedIn.getText());
        user.setBioId(lblUploadPic.getIcon().toString());
        user.setIpAddress(txtIpAddress.getText());
        
        JOptionPane.showMessageDialog(this, "User Information Saved!");
        
        
    }//GEN-LAST:event_btnSubmitUserInfoActionPerformed
    }
    private void txtFNameActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtFNameActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtFNameActionPerformed

    private void txtMobileNumActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtMobileNumActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtMobileNumActionPerformed

    private void txtFaxNumActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtFaxNumActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtFaxNumActionPerformed

    private void txtIpAddressActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtIpAddressActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtIpAddressActionPerformed

    private void txtSocSecNumActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtSocSecNumActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtSocSecNumActionPerformed

    private void txtHealthPlanNumActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtHealthPlanNumActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtHealthPlanNumActionPerformed

    private void txtDeviceIdActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtDeviceIdActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtDeviceIdActionPerformed

    private void btnUploadPhotoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnUploadPhotoActionPerformed
        // TODO add your handling code here:
        JFileChooser browseDriverLicenseFile = new JFileChooser();

 

        //Filter image extensions
        FileNameExtensionFilter fnef = new FileNameExtensionFilter("IMAGES", "png", "jpg", "jpeg");
        browseDriverLicenseFile.addChoosableFileFilter(fnef);
        int showOpenDialogue = browseDriverLicenseFile.showOpenDialog(null);
        if (showOpenDialogue == JFileChooser.APPROVE_OPTION) {
            File selectedImageFile = browseDriverLicenseFile.getSelectedFile();
            String selectedImagePath = selectedImageFile.getAbsolutePath();

 

            //Display image in JLable
            ImageIcon iI = new ImageIcon(selectedImagePath);
            //Resize image to fit jlabel
            Image image = iI.getImage().getScaledInstance(lblUploadPic.getWidth(), lblUploadPic.getHeight(), Image.SCALE_SMOOTH);
            lblUploadPic.setIcon(new ImageIcon(image));
    }//GEN-LAST:event_btnUploadPhotoActionPerformed

    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JToggleButton btnSubmitUserInfo;
    private javax.swing.JButton btnUploadPhoto;
    private javax.swing.JComboBox<String> ddVehType;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JLabel lblAddress;
    private javax.swing.JLabel lblBankAccNum;
    private javax.swing.JLabel lblBioId;
    private javax.swing.JLabel lblBirthDate;
    private javax.swing.JLabel lblCreateUser;
    private javax.swing.JLabel lblDeviceId;
    private javax.swing.JLabel lblEmailAddress;
    private javax.swing.JLabel lblFaxNum;
    private javax.swing.JLabel lblFullName;
    private javax.swing.JLabel lblHealthPlanNum;
    private javax.swing.JLabel lblIpAddress;
    private javax.swing.JLabel lblLicenseNum;
    private javax.swing.JLabel lblLinkedIn;
    private javax.swing.JLabel lblMedRecNum;
    private javax.swing.JLabel lblMobileNum;
    private javax.swing.JLabel lblSocSecNum;
    private javax.swing.JLabel lblUploadPic;
    private javax.swing.JLabel lblVehNumber;
    private javax.swing.JLabel lblVehType;
    private javax.swing.JTextArea txtAddress;
    private javax.swing.JTextField txtBankAccNum;
    private javax.swing.JTextField txtBirthDate;
    private javax.swing.JTextField txtDeviceId;
    private javax.swing.JTextField txtEmailAddress;
    private javax.swing.JTextField txtFName;
    private javax.swing.JTextField txtFaxNum;
    private javax.swing.JTextField txtHealthPlanNum;
    private javax.swing.JTextField txtIpAddress;
    private javax.swing.JTextField txtLicenseNum;
    private javax.swing.JTextField txtLinkedIn;
    private javax.swing.JTextField txtMedRecNum;
    private javax.swing.JTextField txtMobileNum;
    private javax.swing.JTextField txtSocSecNum;
    private javax.swing.JTextField txtVehNum;
    // End of variables declaration//GEN-END:variables
}