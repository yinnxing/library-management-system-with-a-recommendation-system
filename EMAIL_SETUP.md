# Email Notification Setup Guide

## Overview
The library management system now includes automated email notifications for book borrowing and return reminders. This guide explains how to configure and use the email notification features.

## Features

### 1. Book Borrowed Notification
- **Trigger**: When a transaction status changes from PENDING to BORROWED
- **Content**: Book details, due date, borrowing policies, and important reminders
- **Template**: Professional HTML email with library branding

### 2. Return Reminder Notification
- **Trigger**: Automatically sent 3 days before the due date (daily at 9 AM)
- **Content**: Friendly reminder with book details and return instructions
- **Template**: Warning-styled HTML email to grab attention

### 3. Overdue Notification
- **Trigger**: When books become overdue (daily at midnight)
- **Content**: Urgent notice with current overdue fees and immediate action required
- **Template**: Alert-styled HTML email with red color scheme

## Email Configuration

### 1. Gmail SMTP Setup

Update `src/main/resources/application.properties`:

```properties
# Email Configuration
spring.mail.host=smtp.gmail.com
spring.mail.port=587
spring.mail.username=your-email@gmail.com
spring.mail.password=your-app-password
spring.mail.properties.mail.smtp.auth=true
spring.mail.properties.mail.smtp.starttls.enable=true
spring.mail.properties.mail.smtp.starttls.required=true
spring.mail.properties.mail.smtp.ssl.trust=smtp.gmail.com

# Library Email Settings
library.email.from=your-email@gmail.com
library.email.name=Library Management System
```

### 2. Gmail App Password Setup

1. Enable 2-Factor Authentication on your Gmail account
2. Go to Google Account settings → Security → 2-Step Verification
3. Generate an App Password for "Mail"
4. Use this app password in the `spring.mail.password` property

### 3. Alternative Email Providers

#### Outlook/Hotmail
```properties
spring.mail.host=smtp-mail.outlook.com
spring.mail.port=587
spring.mail.username=your-email@outlook.com
spring.mail.password=your-password
```

#### Yahoo Mail
```properties
spring.mail.host=smtp.mail.yahoo.com
spring.mail.port=587
spring.mail.username=your-email@yahoo.com
spring.mail.password=your-app-password
```

## Scheduled Tasks

### 1. Return Reminder Task
- **Schedule**: Daily at 9:00 AM
- **Cron Expression**: `0 0 9 * * ?`
- **Function**: Sends reminders for books due in 3 days

### 2. Overdue Processing Task
- **Schedule**: Daily at midnight
- **Cron Expression**: `0 0 0 * * ?`
- **Function**: Marks books as overdue and sends notifications

## Email Templates

All email templates are responsive HTML with:
- Professional styling
- Library branding
- Clear call-to-action
- Important information highlighting
- Mobile-friendly design

### Template Features:
- **Book Borrowed**: Green theme, congratulatory tone
- **Return Reminder**: Orange theme, friendly reminder
- **Overdue Notice**: Red theme, urgent alert

## Async Processing

Email sending is handled asynchronously to prevent blocking the main application:
- Custom thread pool executor
- 2-5 concurrent threads
- Queue capacity of 100 emails
- Non-blocking email delivery

## Error Handling

- Failed email attempts are logged with detailed error messages
- Email failures don't affect transaction processing
- Retry mechanism can be implemented if needed

## Testing Email Functionality

### 1. Test Email Configuration
```bash
# Check if email service is properly configured
curl -X POST http://localhost:8080/api/transactions/{transactionId}/status/BORROWED
```

### 2. Manual Testing
1. Create a test transaction
2. Update status to BORROWED
3. Check email delivery
4. Verify email content and formatting

### 3. Scheduled Task Testing
```java
// Manually trigger scheduled tasks for testing
@Autowired
private LibraryPolicyService libraryPolicyService;

// Test return reminders
libraryPolicyService.sendReturnReminders();

// Test overdue processing
libraryPolicyService.updateOverdueTransactions();
```

## Troubleshooting

### Common Issues:

1. **Authentication Failed**
   - Verify app password is correct
   - Ensure 2FA is enabled for Gmail
   - Check username format

2. **Connection Timeout**
   - Verify SMTP host and port
   - Check firewall settings
   - Ensure internet connectivity

3. **Emails Not Sending**
   - Check application logs for errors
   - Verify email addresses are valid
   - Ensure async processing is enabled

4. **HTML Not Rendering**
   - Check email client HTML support
   - Verify MIME type is set correctly
   - Test with different email clients

### Debug Logging
Add to `application.properties`:
```properties
logging.level.com.library.booksystem.service.EmailService=DEBUG
logging.level.org.springframework.mail=DEBUG
```

## Security Considerations

1. **Never commit email credentials** to version control
2. Use environment variables for production:
   ```bash
   export SPRING_MAIL_USERNAME=your-email@gmail.com
   export SPRING_MAIL_PASSWORD=your-app-password
   ```
3. Consider using OAuth2 for production environments
4. Implement rate limiting for email sending
5. Use encrypted connections (TLS/SSL)

## Production Deployment

### Environment Variables
```bash
# Email Configuration
SPRING_MAIL_HOST=smtp.gmail.com
SPRING_MAIL_PORT=587
SPRING_MAIL_USERNAME=${EMAIL_USERNAME}
SPRING_MAIL_PASSWORD=${EMAIL_PASSWORD}
LIBRARY_EMAIL_FROM=${EMAIL_USERNAME}
LIBRARY_EMAIL_NAME="Your Library Name"
```

### Docker Configuration
```dockerfile
ENV SPRING_MAIL_USERNAME=${EMAIL_USERNAME}
ENV SPRING_MAIL_PASSWORD=${EMAIL_PASSWORD}
ENV LIBRARY_EMAIL_FROM=${EMAIL_USERNAME}
```

## Monitoring and Analytics

Consider implementing:
- Email delivery tracking
- Open rate monitoring
- Click-through rate analysis
- Bounce rate tracking
- User engagement metrics

## Future Enhancements

Potential improvements:
- Email templates customization
- Multi-language support
- SMS notifications
- Push notifications
- Email preferences management
- Delivery status tracking
- A/B testing for email content 